"""
Helper functions for Hollow Space Carving (HSC).

This module provides utilities to generate various types of unknown
features required by the HSC approach. Specifically, it includes:

* **Extrapolation Mix (ExtrapMix)** for creating near-boundary unknown
  features by randomly extrapolating pairs of known support features.
* **Energy Diffusion** to push features away from their prototypes
  along the gradient of an energy head, producing medium-distance
  unknowns.
* **Far OOD Sampling** to sample features from an out-of-distribution
  dataset via a given encoder.

These functions are designed to be modular and can be invoked from
within the model or trainer without tight coupling to the rest of the
codebase. All returned features are normalised to lie on the unit
hypersphere, matching the assumptions of the HSC model.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def extrap_mix(
    features: torch.Tensor,
    lam_range: Tuple[float, float] = (1.0, 2.0),
    num_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate near-boundary unknown features via extrapolation.

    Given a collection of feature vectors (typically support set
    embeddings), this function randomly selects pairs of distinct
    features and creates new features by extrapolating one feature away
    from the other. The extrapolation coefficient ``λ`` is sampled
    uniformly from ``lam_range``. Each extrapolated feature ``x_new``
    is computed as ``x_i + λ * (x_i - x_j)`` and then L2-normalised.

    Parameters
    ----------
    features : torch.Tensor of shape ``(N, D)``
        The input feature matrix containing ``N`` feature vectors of
        dimension ``D``. The features are assumed to be L2-normalised.
    lam_range : (float, float), optional
        A tuple specifying the inclusive range from which the
        extrapolation coefficient ``λ`` is sampled. Values greater
        than 1.0 push the new feature further away from ``x_j``.
    num_samples : int, optional
        The number of new samples to generate. If ``None``, the
        function will generate ``N`` new features.
    device : torch.device, optional
        Device on which to allocate the returned tensor. If ``None``,
        defaults to the device of ``features``.

    Returns
    -------
    torch.Tensor of shape ``(num_samples, D)``
        A tensor containing the extrapolated and normalised feature
        vectors.
    """
    if num_samples is None:
        num_samples = features.size(0)
    n, d = features.size()
    device = device or features.device

    # Prepare container for new features
    new_features = torch.empty(num_samples, d, device=device)
    for idx in range(num_samples):
        # Randomly pick two distinct indices
        i = random.randrange(n)
        j = random.randrange(n - 1)
        # ensure j != i
        if j >= i:
            j += 1
        x_i = features[i]
        x_j = features[j]
        lam = random.uniform(*lam_range)
        # extrapolate away from x_j towards x_i
        x_new = x_i + lam * (x_i - x_j)
        # normalise to unit sphere
        x_new = F.normalize(x_new.unsqueeze(0), dim=1).squeeze(0)
        new_features[idx] = x_new
    return new_features


@torch.no_grad()
def sample_far_ood(
    loader: Iterable,
    encoder: nn.Module,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample far out-of-distribution features using a given encoder.

    This helper iterates over a dataloader providing OOD images,
    extracts their features via the supplied ``encoder`` (assumed to
    output unnormalised embeddings), normalises them, and returns the
    first ``num_samples`` features collected. It is intended for use
    during training to provide far OOD examples for the HSC loss.

    Parameters
    ----------
    loader : Iterable
        An iterable yielding batches of images (and optionally
        labels). Only the images are consumed.
    encoder : nn.Module
        A neural network used to extract feature embeddings from
        images. It should be set to evaluation mode prior to calling
        this function.
    num_samples : int
        The desired number of OOD feature vectors to return.
    device : torch.device, optional
        Device on which to perform computation and store the results.

    Returns
    -------
    torch.Tensor of shape ``(num_samples, D)``
        A tensor containing ``num_samples`` normalised OOD features.
    """
    # Determine device from encoder if not provided
    encoder_device = next(encoder.parameters()).device
    device = device or encoder_device
    collected = []
    for batch in loader:
        # assume batch is either (images, labels) or just images
        if isinstance(batch, Sequence) or isinstance(batch, tuple):
            images = batch[0]
        else:
            images = batch
        images = images.to(device)
        # forward pass through encoder
        feats = encoder(images)
        # flatten and normalise
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        feats = F.normalize(feats, dim=1)
        collected.append(feats.cpu())
        if sum(x.size(0) for x in collected) >= num_samples:
            break
    if not collected:
        raise RuntimeError('No OOD features could be sampled from the loader.')
    feats = torch.cat(collected, dim=0)[:num_samples]
    return feats.to(device)


def energy_diffusion(
    features: torch.Tensor,
    energy_head: nn.Module,
    steps: int = 5,
    step_size: float = 0.1,
    normalise_each_step: bool = True,
    detach_input: bool = False,
) -> torch.Tensor:
    """Generate medium-distance unknowns by ascending the energy gradient.

    This function performs iterative gradient ascent on the input
    features with respect to an energy head. At each step it computes
    the gradient of the summed energy output with respect to the
    features, updates the features by a scaled step in the direction
    of the gradient, and optionally normalises the features to the
    unit sphere. The resulting features are detached from the
    computation graph before being returned.

    Parameters
    ----------
    features : torch.Tensor of shape ``(N, D)``
        The initial feature vectors on which to perform diffusion. The
        tensor must be float and will be modified; a detached copy
        will be returned. If ``detach_input`` is ``True``, a cloned
        detached version of ``features`` will be used as the starting
        point.
    energy_head : nn.Module
        A neural network that takes a feature tensor of shape ``(N, D)``
        and outputs a tensor of energies of shape ``(N, 1)`` or ``(N,)``.
    steps : int, optional
        The number of gradient ascent steps to perform. More steps push
        the features further away from their original positions.
    step_size : float, optional
        The step size (learning rate) for gradient ascent.
    normalise_each_step : bool, optional
        Whether to normalise the features to the unit sphere after each
        update. Enabling this keeps the features on the sphere and
        stabilises the updates.
    detach_input : bool, optional
        If ``True``, the input features will be cloned and detached
        before starting diffusion. When ``False``, the updates will
        operate on the provided tensor directly and its gradient
        history will be preserved.

    Returns
    -------
    torch.Tensor of shape ``(N, D)``
        The resulting diffused features, detached from the graph.
    """
    # Ensure we work on a tensor that has requires_grad set
    x = features
    if detach_input:
        x = x.clone().detach()
    x.requires_grad_(True)
    for _ in range(steps):
        # Forward through energy head
        energy = energy_head(x)
        # Flatten to shape (N,) for summation; assume energy returns (N, 1) or (N,)
        if energy.dim() > 1:
            energy = energy.squeeze(-1)
        # We want to ascend energy, so compute negative loss
        loss = -energy.sum()
        # Backprop to get gradient w.r.t. x
        loss.backward()
        grad = x.grad
        # Update features
        with torch.no_grad():
            x.add_(step_size * grad)
            if normalise_each_step:
                x.copy_(F.normalize(x, dim=1))
        # Clear gradient for next iteration
        x.grad.zero_()
    # Detach and normalise one final time
    with torch.no_grad():
        x_final = F.normalize(x, dim=1)
    return x_final.detach()


__all__ = ['extrap_mix', 'sample_far_ood', 'energy_diffusion']