"""
Hollow Space Carving (HSC) model implementation.

This module defines ``HSCModel``, a few-shot open‑set recognition model
that augments a standard feature encoder with energy and evidence heads
and introduces a simple metric‑based classifier on a hypersphere. It is
designed to be used within the Meta Evidential Transformer (MET)
codebase, replacing the FEAT head for open‑set few‑shot tasks.

Key components:
  * **Prototypical classifier:** prototypes are the mean of support
    embeddings per class, normalised to lie on the unit hypersphere.
    Classification logits for query samples are computed via scaled
    cosine similarity.
  * **Energy head:** a lightweight MLP producing a single energy
    scalar for each query embedding. During training this output is
    used in the energy shaping loss to enforce different energy
    levels for known and unknown samples.
  * **Evidence head:** another MLP producing non‑negative Dirichlet
    evidence for each known class. This is used in combination with
    the evidential loss to model epistemic uncertainty.
  * **Unknown generation:** a helper method ``generate_unknowns``
    (optionally) produces near‑boundary, diffused and far OOD
    features from support embeddings. These can be supplied to the
    trainer to form unknown samples during training.

The model returns additional outputs (energies and evidences) beyond
the classification logits. Trainers using this class should be
modified accordingly to handle these values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models import FewShotModel
from model.utils import hsc_utils


class HSCModel(FewShotModel):
    """Hollow Space Carving model for few‑shot open‑set recognition.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration options. The following attributes are expected:
        ``closed_way`` (int), ``open_way`` (int, optional),
        ``shot`` (int), ``query`` (int), ``backbone_class`` (str),
        as well as optional hyperparameters ``kappa`` (float) and
        ``open_margin`` (float). Additional attributes such as
        ``diffusion_steps`` may be used for unknown generation.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # Determine the dimension of embeddings produced by the backbone.
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone_class}")
        self.hdim = hdim

        # Hyperparameters for cosine classifier on the hypersphere.
        # kappa scales the cosine similarity; margin enforces an open gap.
        self.kappa = float(getattr(args, 'kappa', 10.0))
        self.margin = float(getattr(args, 'open_margin', 0.1))

        # Hidden dimension for the heads. If not specified, use a simple rule.
        hidden_dim = getattr(args, 'hidden_dim', max(hdim // 2, 128))

        # Energy head: maps from embedding dimension to a single scalar.
        # Use a small MLP with a ReLU in the middle.
        self.energy_head = nn.Sequential(
            nn.Linear(hdim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Evidence head: outputs non‑negative evidence for each known class.
        # A softplus activation ensures positivity. The output dimension
        # equals the number of closed (known) classes.
        self.evidence_head = nn.Sequential(
            nn.Linear(hdim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, args.closed_way),
            nn.Softplus()
        )

        # Optionally set by the trainer to supply a DataLoader yielding
        # far OOD images for unknown generation.
        self.ood_loader = None

        # Number of diffusion steps for energy diffusion unknowns.
        self.diffusion_steps = int(getattr(args, 'diffusion_steps', 3))

    def forward(self, x, get_feature: bool = False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (1, N, C, H, W).
        get_feature : bool, optional
            If True, return raw embeddings from the encoder without
            performing classification. In this case the return value is
            identical to ``self.encoder(x)``. This flag is used by the
            trainer during evaluation.

        Returns
        -------
        tuple
            During training, returns a tuple containing classification
            logits for known queries, classification logits for open
            queries (if ``args.open_loss`` is True), a placeholder for
            regularisation logits (``None`` by default), a tensor of
            energies and a tensor of Dirichlet evidences. During
            evaluation the tuple contains classification logits,
            energies and evidences only.
        """
        if get_feature:
            return super().forward(x, get_feature=True)

        # Squeeze the dummy batch dimension inserted by the DataLoader.
        x = x.squeeze(0)
        # Extract instance embeddings.
        instance_embs = self.encoder(x)
        # Determine support and query indices.
        support_idx, query_idx = self.split_instances(x)

        if self.training:
            return self._forward(instance_embs, support_idx, query_idx)
        else:
            # In evaluation, only return classification and uncertainty.
            out = self._forward(instance_embs, support_idx, query_idx)
            return out['close_logits'], out['energies'], out['evidence']

    def _forward(self, instance_embs: torch.Tensor,
                 support_idx: torch.Tensor,
                 query_idx: torch.Tensor):
        """Internal forward method computing logits, energies and evidences.

        This method separates support and query instances, computes
        class prototypes from support embeddings, normalises them to the
        unit hypersphere, and then computes cosine similarities between
        query features and prototypes. It also applies the energy and
        evidence heads to query features.

        Returns
        -------
        tuple
            In training mode with open_set tasks enabled
            (``args.open_loss`` is True), returns
            ``(close_logits, open_logits, logits_reg, energies, evidence)``.
            When not using open set tasks, returns
            ``(close_logits, logits_reg, energies, evidence)``.
            In evaluation mode, returns ``(logits, energies, evidence)``.
        """
        args = self.args
        emb_dim = instance_embs.size(-1)

        # Gather support and query embeddings using the provided indices.
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))

        # Restrict to closed classes on support (ignore any open classes).
        support_close = support[:, :, :args.closed_way, :]

        # Compute class prototypes by averaging support embeddings over shots.
        proto = support_close.mean(dim=1)  # shape: (1, closed_way, emb_dim)
        # Normalise prototypes to lie on the unit hypersphere.
        proto = F.normalize(proto, dim=-1)
        # Flatten prototype tensor for similarity computation.
        proto_norm = proto.squeeze(0)  # (closed_way, emb_dim)

        # Separate query embeddings into close (known) and open sets.
        if self.training and args.open_loss:
            query_close = query[:, :, :args.closed_way, :]
            query_open = query[:, :, args.closed_way:, :]
        else:
            query_close = query
            query_open = None

        # Flatten query embeddings and normalise them.
        close_feat = query_close.reshape(-1, emb_dim)
        close_feat_norm = F.normalize(close_feat, dim=-1)
        # Compute scaled cosine similarities to prototypes.
        close_logits = torch.matmul(close_feat_norm, proto_norm.t())  # (n_close, closed_way)
        close_logits = self.kappa * close_logits - self.margin

        # Compute energy and evidence for known queries.
        energies_close = self.energy_head(close_feat).squeeze(-1)
        evidences_close = self.evidence_head(close_feat)

        # Handle open queries if present.
        if query_open is not None:
            open_feat = query_open.reshape(-1, emb_dim)
            open_feat_norm = F.normalize(open_feat, dim=-1)
            open_logits = torch.matmul(open_feat_norm, proto_norm.t())  # (n_open, closed_way)
            open_logits = self.kappa * open_logits - self.margin

            energies_open = self.energy_head(open_feat).squeeze(-1)
            evidences_open = self.evidence_head(open_feat)

            # Concatenate energies and evidences across close and open queries.
            energies = torch.cat([energies_close, energies_open], dim=0)
            evidence = torch.cat([evidences_close, evidences_open], dim=0)
        else:
            open_logits = None
            energies = energies_close
            evidence = evidences_close

        # logits_reg is unused in HSC; return None for compatibility.
        logits_reg = None

        # 真实未知证据：用支持集生成未知特征 → evidence_unknown =====
        evidence_unknown = None
        unk_energies = None

        if self.training:
            # 用你已经写好的 generate_unknowns（它需要 support_feats 形状: [B, shot, closed_way, D]）
            # 你此处的 support_close 形状正好满足
            hu_feats, diff_feats, far_feats = self.generate_unknowns(support_close)  # 各为 [N?, D] 或 None

            feats_list = [t for t in (hu_feats, diff_feats, far_feats) if t is not None and t.numel() > 0]
            if len(feats_list) > 0:
                unk_feats = torch.cat(feats_list, dim=0)  # [Nu, D]

                # 未知能量（可选，用于能量整形的“未知高能”）
                unk_feats_detached = unk_feats.detach()  # 不回传到 encoder
                unk_energies = self.energy_head(unk_feats_detached).squeeze(-1)


                # 关键：未知证据 —— 让梯度更新 evidence_head，但不回流到 encoder（更稳）
                # 你的 evidence_head 末尾是 Softplus，输出本身就是非负的 evidence（而不是 alpha），可以直接用
                evidence_unknown = self.evidence_head(unk_feats.detach())  # [Nu, C], evidence >= 0

        if self.training:
            # 建议统一返回 dict，便于 trainer 使用 evidence_unknown/unk_energies
            out = {
                'close_logits': close_logits,      # [Bq, C]
                'energies': energies,              # [Bq]
                'evidence': evidence,              # [Bq, C]（闭集证据）
                'open_logits': open_logits,        # [Bo, C] 或 None
                'logits_reg': logits_reg,          # None（兼容旧接口）
                'unk_energies': unk_energies,      # [Nu] 或 None
                'evidence_unknown': evidence_unknown,  # [Nu, C] 或 None
            }
            return out
        else:
            # 评估阶段的最小返回（也可同样返回 dict，一致性更好）
            out = {
                'close_logits': close_logits,
                'energies': energies,
                'evidence': evidence,
            }
            return out

    def generate_unknowns(self, support_feats: torch.Tensor):
        """Generate unknown feature embeddings for training.

        This convenience method uses the utilities in ``hsc_utils`` to
        produce near‑boundary (HU), diffused (middle) and far OOD
        embeddings from a set of support embeddings. The number of
        generated unknowns matches the number of support embeddings.

        Parameters
        ----------
        support_feats : torch.Tensor of shape (1, shot, closed_way, D)
            Support set features extracted from the encoder. The tensor
            should already be normalised if the subsequent classifier
            operates on the unit hypersphere.

        Returns
        -------
        tuple of torch.Tensor
            A tuple ``(hu_feats, diff_feats, far_feats)`` where each
            element is a tensor of shape (num_support, D). ``far_feats``
            will be ``None`` if ``self.ood_loader`` is not set.
        """
        # Collapse support features to a flat matrix of embeddings.
        B, shot, n_cls, dim = support_feats.shape
        sup = support_feats.view(-1, dim)
        # Ensure features are normalised.
        sup = F.normalize(sup, dim=1)

        # Generate near‑boundary unknowns via extrapolation mix.
        hu_feats = hsc_utils.extrap_mix(sup, num_samples=sup.size(0), device=sup.device)

        # Generate medium unknowns via energy diffusion.
        diff_feats = hsc_utils.energy_diffusion(
            sup,
            self.energy_head,
            steps=self.diffusion_steps,
            step_size=0.1,
            normalise_each_step=True,
            detach_input=True,
        )

        # Generate far OOD unknowns if an OOD loader is provided.

        far_feats = self.generate_far_ood_samples(
            num_samples=sup.size(0),
            image_size=84,
            device=sup.device,
        )

        return hu_feats, diff_feats, far_feats

def generate_far_ood_samples(self, num_samples, image_size, device):
    """Generate far OOD (Out-of-Distribution) samples."""
    # 使用高斯噪声生成远域样本（可以改为其他生成方法）
    noise = torch.randn(num_samples, 3, image_size, image_size, device=device)  # 高斯噪声

    far_ood_samples = noise  #

    return far_ood_samples


__all__ = ['HSCModel']