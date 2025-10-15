"""
Loss functions for Hollow Space Carving (HSC).

This module defines a set of loss functions used by the HSC model to
encourage tight clustering of closed-set features on a hypersphere,
shape energy outputs for different categories of data, and regularise
the evidential outputs for unknown samples. These losses are designed
to work together with the MET framework's evidential learning
components.

Functions
---------
angle_compact_loss(features, prototypes, labels, margin, kappa)
    Computes a von Mises–Fisher inspired angular loss that pulls
    features towards their class prototypes while imposing an open-set
    margin.

energy_shaping_loss(energies, targets, huber, delta)
    Penalises deviations between predicted energies and desired target
    energies for different categories of samples using either MSE or
    Huber loss.

kl_unknown_loss(evidence, num_classes, reduction)
    Computes the KL divergence between the predicted Dirichlet
    distribution for unknown samples and a uniform Dirichlet prior.
"""

import torch
import torch.nn.functional as F


def angle_compact_loss(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    kappa: float = 10.0,
    logits: torch.Tensor = None,
) -> torch.Tensor:
    """Compute an angular compactness loss with an open-set margin.

    This loss encourages each feature vector to align closely with its
    corresponding class prototype while maintaining an angular margin
    between classes. Conceptually, it implements a von Mises–Fisher
    (vMF) likelihood for classification on the unit hypersphere. A
    larger concentration parameter ``kappa`` makes the loss steeper,
    pulling features more strongly towards their prototypes. The
    margin ``margin`` is subtracted from the similarity of the correct
    class, effectively requiring that the feature be closer to its
    prototype by at least that margin compared with other classes.

    Parameters
    ----------
    features : torch.Tensor, shape ``(N, D)``
        Normalised feature vectors of the query/support set.
    prototypes : torch.Tensor, shape ``(C, D)``
        Normalised class prototypes on the unit hypersphere.
    labels : torch.Tensor, shape ``(N,)``
        Ground-truth integer labels for each feature. Values should
        range from ``0`` to ``C-1``.
    margin : float, optional
        Angular margin subtracted from the similarity of the correct
        class. A positive margin enforces a separation between closed
        classes and the open space.
    kappa : float, optional
        Concentration parameter controlling the sharpness of the
        angular distribution. Higher values yield a more peaked loss.

    Returns
    -------
    torch.Tensor
        A scalar loss tensor representing the average angular loss
        across all samples.
    """
    if logits is not None:
        return F.cross_entropy(logits, labels)
    # Compute cosine similarities between features and prototypes
    # Expect both features and prototypes to be L2-normalised
    logits = F.linear(features, prototypes)  # shape (N, C)
    # Gather the cosine similarity of the correct class for each sample
    target_cos = logits.gather(1, labels.view(-1, 1))
    # Apply margin: subtract margin from the correct class similarity
    # We clone logits to avoid modifying the original tensor in-place
    logits_with_margin = logits * 1.0
    logits_with_margin.scatter_(1, labels.view(-1, 1), target_cos - margin)
    # Scale logits by kappa to control concentration
    scaled_logits = kappa * logits_with_margin
    # Use cross-entropy loss over the scaled logits
    loss = F.cross_entropy(scaled_logits, labels)
    return loss


def energy_shaping_loss(
    energies: torch.Tensor,
    targets: torch.Tensor,
    huber: bool = False,
    delta: float = 0.1,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Compute a loss that shapes energies towards target values.

    Energy outputs from the model are used to distinguish between
    closed-set samples and various types of unknowns. This loss
    penalises the squared (or Huber) difference between each energy and
    its desired target energy. For example, closed-set samples might
    have a low target energy (e.g., 0.1) while near and far unknowns
    have progressively higher target energies.

    Parameters
    ----------
    energies : torch.Tensor, shape ``(N,)`` or ``(N, 1)``
        Predicted energy values for each sample.
    targets : torch.Tensor, shape ``(N,)`` or ``(N, 1)``
        The target energy values corresponding to each sample. Typically
        provided as a tensor of the same shape as ``energies``.
    huber : bool, optional
        If ``True``, use the Huber loss with parameter ``delta``.
        Otherwise, use mean squared error (MSE).
    delta : float, optional
        The threshold at which the Huber loss transitions from
        quadratic to linear. Ignored when ``huber=False``.
    reduction : str, optional
        Specifies the reduction to apply to the output: ``'mean'`` |
        ``'sum'``. Default: ``'mean'``.

    Returns
    -------
    torch.Tensor
        A scalar loss tensor representing the aggregated energy shaping
        loss.
    """
    # Flatten energies and targets to shape (N,)
    energies = energies.view(-1)
    targets = targets.view(-1).to(energies.device)
    diff = energies - targets
    if huber:
        # Huber loss: quadratic for small errors, linear for large errors
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = delta * (abs_diff - 0.5 * delta)
        per_elem = torch.where(abs_diff < delta, quadratic, linear)
    else:
        # Mean squared error
        per_elem = diff.pow(2)
    if reduction == 'mean':
        return per_elem.mean()
    elif reduction == 'sum':
        return per_elem.sum()
    else:
        # No reduction
        return per_elem


def kl_unknown_loss(
    evidence: torch.Tensor,
    num_classes: int,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Compute the KL divergence between a Dirichlet posterior and the uniform prior.

    For unknown samples, we want the posterior Dirichlet distribution to
    approach a uniform distribution over classes, indicating maximal
    uncertainty. This loss computes the KL divergence from the
    predicted Dirichlet (parameterised by evidence) to a uniform
    Dirichlet prior with parameters equal to 1 for each class.

    Parameters
    ----------
    evidence : torch.Tensor, shape ``(N, C)``
        Evidence logits produced by the model's evidence head. The
        Dirichlet concentration parameters are computed as
        ``alpha = relu(evidence) + 1``.
    num_classes : int
        Number of classes (C). Must match the second dimension of
        ``evidence``.
    reduction : str, optional
        Specifies the reduction to apply to the output: ``'mean'`` |
        ``'sum'`` |
        ``'none'``. Default: ``'mean'``.

    Returns
    -------
    torch.Tensor
        The KL divergence loss. If ``reduction`` is ``'none'``, returns
        a tensor of shape ``(N,)`` containing per-sample divergences.
    """
    if evidence.numel() == 0:
        return torch.tensor(0.0, device=evidence.device)
    # Compute Dirichlet parameters
    alpha = F.relu(evidence) + 1.0
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    # Uniform prior parameters (all ones)
    ones = torch.ones(1, num_classes, device=evidence.device, dtype=evidence.dtype)
    # KL divergence between Dirichlet distributions
    # term1 = log Gamma(sum alpha) - sum log Gamma(alpha)
    term1 = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    # term2 = sum log Gamma(ones) - log Gamma(sum ones)
    term2 = torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    # term3 = (alpha - ones) * (digamma(alpha) - digamma(sum alpha))
    term3 = (alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    kl = term1 + term2 + term3.sum(dim=1, keepdim=True)
    kl = kl.squeeze(1)
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


__all__ = [
    'angle_compact_loss',
    'energy_shaping_loss',
    'kl_unknown_loss',
]