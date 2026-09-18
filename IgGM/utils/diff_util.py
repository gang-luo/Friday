"""Utility functions.

Notes:
* All the interpolation routines are used to calculate the expectation of $x_{t-1}$, which is fully
    deterministic. For DDPM sampling, random noise should be further applied.
"""
import os
import random
import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.constraints import simplex
from torch.distributions.categorical import Categorical
from torch.distributions import Distribution, constraints, Normal, MultivariateNormal

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C, N_ATOMS_PER_RESD, ATOM_INFOS_PER_RESD, RESD_MAP_1TO3
from IgGM.transform.so3 import vec2skew, skew2vec
from math import pi, sqrt

def cord2fram_batch(cord_tns):
    """Convert 3D coordinates into local frames in the batch mode.

    Args:
    * cord_tns: 3D coordinates of size N x 3 x 3

    Returns:
    * rota_tns: rotation matrices of size N x 3 x 3
    * trsl_mat: translation vectors of size N x 3
    """

    eps = 1e-6
    x1, x2, x3 = [x.squeeze(dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    u2 = v2 - torch.sum(e1 * v2, dim=1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=1)
    rota_tns = torch.stack([e1, e2, e3], dim=1).permute(0, 2, 1)
    trsl_mat = x2

    return rota_tns, trsl_mat

def ss2ptr(aa_seqs, cord_tns, cmsk_tns):
    """Convert amino-acid sequences & per-atom 3D coordinates into Prob-Trsl-Rota parameters.

    Args:
    * aa_seqs: list of amino-acid sequences, each of length L
    * cord_tns: batched per-atom 3D coordinates of size N x L x M x 3
    * cmsk_tns: batched per-atom 3D coordinates' validness masks of size N x L x M

    Returns:
    * prob_tns: batched probabilistic distributions of size N x L x C
    * trsl_tns: batched translation vectors of size N x L x 3
    * rota_tns: batched rotation matrices of size N x L x 3 x 3
    * fmsk_mat: batched per-residue local frames' validness masks of size N x L
    """

    # initialization
    n_smpls = len(aa_seqs)
    n_resds = len(aa_seqs[0])
    device = cord_tns.device
    n_atyps = len(RESD_NAMES_1C)

    # flatten amino-acid sequences & per-atom 3D coordinates
    atom_mapper = AtomMapper()
    aa_seq_flat = ''.join(aa_seqs)
    if 'X' in aa_seq_flat:
        aa_seq_flat = aa_seq_flat.replace('X', 'T')  # replace 'X' w/ 'A'
    cord_tns_fa_flat = cord_tns.view(n_smpls * n_resds, N_ATOMS_PER_RESD, 3)
    cmsk_mat_fa_flat = cmsk_tns.view(n_smpls * n_resds, N_ATOMS_PER_RESD)
    cord_tns_bb_flat = atom_mapper.run(aa_seq_flat, cord_tns_fa_flat, frmt_src='n14-tf', frmt_dst='n3')
    cmsk_mat_bb_flat = atom_mapper.run(aa_seq_flat, cmsk_mat_fa_flat, frmt_src='n14-tf', frmt_dst='n3')

    # convert amino-acid sequences into probabilistic distributions
    tokn_vec_flat = torch.tensor([RESD_NAMES_1C.index(x) for x in aa_seq_flat], dtype=torch.int64)
    prob_mat_flat = nn.functional.one_hot(tokn_vec_flat, num_classes=n_atyps).to(torch.float32)
    prob_tns = prob_mat_flat.view(n_smpls, n_resds, n_atyps).to(device)

    # convert per-atom 3D coordinates into translation vectors & rotation matrices
    rota_tns_flat, trsl_mat_flat = cord2fram_batch(cord_tns_bb_flat)
    fmsk_vec_flat = torch.all(cmsk_mat_bb_flat == 1, dim=1)
    trsl_mat_flat = torch.where(
        fmsk_vec_flat.view(-1, 1), trsl_mat_flat, torch.zeros(3, device=device).unsqueeze(dim=0))
    rota_tns_flat = torch.where(
        fmsk_vec_flat.view(-1, 1, 1), rota_tns_flat, torch.eye(3, device=device).unsqueeze(dim=0))
    trsl_tns = trsl_mat_flat.view(n_smpls, n_resds, 3)
    rota_tns = rota_tns_flat.view(n_smpls, n_resds, 3, 3)
    fmsk_mat = fmsk_vec_flat.to(torch.int8).view(n_smpls, n_resds)

    return prob_tns, trsl_tns, rota_tns, fmsk_mat


def ptr2ss(prob_tns, trsl_tns, rota_tns, fmsk_mat, stoc_seq=False):
    """Convert Prob-Trsl-Rota parameters into amino-acid sequences & per-atom 3D coordinates.

    Args:
    * prob_tns: batched probabilistic distributions of size N x L x C
    * trsl_tns: batched translation vectors of size N x L x 3
    * rota_tns: batched rotation matrices of size N x L x 3 x 3
    * fmsk_mat: batched per-residue local frames' validness masks of size N x L
    * stoc_seq: (optional) whether amino-acid sequences are stochastically determined

    Returns:
    * aa_seqs: list of amino-acid sequences, each of length L
    * cord_tns: batched per-atom 3D coordinates of size N x L x M x 3
    * cmsk_tns: batched per-atom 3D coordinates' validness masks of size N x L x M
    """

    # initialization
    device = prob_tns.device
    n_smpls = prob_tns.shape[0]
    n_resds = prob_tns.shape[1]

    # obtain reference 3D coordinates for backbone atoms
    dcrd_mat_bb_dict = {}
    for resd_name in RESD_NAMES_1C:
        atom_infos = ATOM_INFOS_PER_RESD[RESD_MAP_1TO3[resd_name]][:3]  # N - CA - C
        # assert [x[0] for x in atom_infos] == ['N', 'CA', 'C']  # validate atom names
        dcrd_mat_bb_dict[resd_name] = torch.tensor([x[2] for x in atom_infos], dtype=torch.float32)

    # convert probabilistic distributions into amino-acid sequences
    if stoc_seq:
        distr = Categorical(probs=prob_tns)
        ridx_mat = distr.sample()  # N x L
    else:
        ridx_mat = torch.argmax(prob_tns, dim=2)
    ridx_mat_np = ridx_mat.detach().cpu().numpy()
    aa_seqs = [''.join(RESD_NAMES_1C[x] for x in ridx_mat_np[idx]) for idx in range(n_smpls)]
    aa_seq_flat = ''.join(aa_seqs)

    # convert translation vectors & rotation matrices into per-atom 3D coordinates
    atom_mapper = AtomMapper()
    dcrd_tns_bb_flat = torch.stack([dcrd_mat_bb_dict[x] for x in aa_seq_flat], dim=0).to(device)
    cord_tns_bb_flat = trsl_tns.view(n_smpls * n_resds, 1, 3) + \
                       torch.bmm(dcrd_tns_bb_flat, rota_tns.view(n_smpls * n_resds, 3, 3).permute(0, 2, 1))
    cmsk_mat_bb_flat = fmsk_mat.view(n_smpls * n_resds, 1).repeat(1, 3)
    cord_tns_bb_flat *= cmsk_mat_bb_flat.unsqueeze(dim=-1)  # mask-out invalid 3D coordinates
    cord_tns_fa_flat = atom_mapper.run(aa_seq_flat, cord_tns_bb_flat, frmt_src='n3', frmt_dst='n14-tf')
    cmsk_mat_fa_flat = atom_mapper.run(aa_seq_flat, cmsk_mat_bb_flat, frmt_src='n3', frmt_dst='n14-tf')
    cord_tns = cord_tns_fa_flat.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
    cmsk_tns = cmsk_mat_fa_flat.view(n_smpls, n_resds, N_ATOMS_PER_RESD)

    return aa_seqs, cord_tns, cmsk_tns


def prob2seq(prob_tns, stoc_seq=False):
    """Convert residue-type probabilities into amino-acid sequences.

    Args:
    * prob_tns: probabilistic distributions of size N x L x C
    * stoc_seq: whether to sample stochastically instead of argmax decoding

    Returns:
    * aa_seqs: list of amino-acid sequences, each of length L
    """

    if stoc_seq:
        distr = Categorical(probs=prob_tns)
        ridx_mat = distr.sample()
    else:
        ridx_mat = torch.argmax(prob_tns, dim=2)
    ridx_mat_np = ridx_mat.detach().cpu().numpy()
    return [''.join(RESD_NAMES_1C[x] for x in ridx_mat_np[idx]) for idx in range(ridx_mat.shape[0])]


def calc_intp_coeffs(alpha_bar_prev, alpha_bar_curr, has_noise):
    """Calculate interpolation coefficients.

    Args:
    * alpha_bar_prev: previous accumulated alpha coefficient ($\bar{\alpha}_{t-1}$)
    * alpha_bar_curr: current accumulated alpha coefficient ($\bar{\alpha}_{t}$)
    * has_noise: whether random noise is enabled (DDPM) or not (DDIM)

    Returns:
    * theta: weighting coefficient for the initial time-step ($x_{0}$)
    * gamma: weighting coefficient for the current time-step ($x_{t}$)
    """

    alpha_curr = alpha_bar_curr / alpha_bar_prev
    if has_noise:
        sigma_sqr = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr) * (1.0 - alpha_curr)
    else:
        sigma_sqr = torch.zeros_like(alpha_bar_prev)
    gamma = torch.sqrt((1.0 - alpha_bar_prev - sigma_sqr) / (1.0 - alpha_bar_curr))  # x_t
    theta = torch.sqrt(alpha_bar_prev) - gamma * torch.sqrt(alpha_bar_curr)  # x_0

    return theta, gamma


def intp_prob_mat_dsct(prob_mat_init, prob_mat_curr, trmat_ac_prev, trmat_ac_curr):
    """Interpolate between $x_0$ (prot_mat_init) and $x_{t}$ (prot_mat_curr) - discrete.

    Args:
    * prob_mat_init: initial probabilistic distributions ($x_{0}$) of size N x C
    * prob_mat_curr: current probabilistic distributions ($x_{t}$) of size N x C
    * trmat_ac_prev: previous accumulated transition matrix ($\bar{Q}_{t-1}$) of size C x C
    * trmat_ac_curr: current accumulated transition matrix ($\bar{Q}_{t}$) of size C x C

    Returns:
    * prob_mat_prev: previous probabilistic distributions ($x_{t - 1}$) of size N x C

    Reference:
    * Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces.
    """

    # validate input tensors
    assert prob_mat_init.ndim == 2
    n_smpls, n_types = prob_mat_init.shape
    assert list(prob_mat_curr.shape) == [n_smpls, n_types]
    assert list(trmat_ac_prev.shape) == [n_types, n_types]
    assert list(trmat_ac_curr.shape) == [n_types, n_types]
    assert torch.all(simplex.check(prob_mat_init))
    assert torch.all(simplex.check(prob_mat_curr))

    # obtain the single-step transition matrix
    trmat_st_curr = torch.linalg.solve(trmat_ac_prev, trmat_ac_curr)

    # calculate probabilistic distributions for $x_{t - 1}$
    is_onht = torch.all(torch.max(prob_mat_init, dim=1)[0] == 1.0)
    prob_mat_pri = torch.sum(prob_mat_curr.unsqueeze(dim=2) * trmat_st_curr.T.unsqueeze(dim=0), dim=1)
    if is_onht:
        prob_mat_sec = torch.sum(prob_mat_init.unsqueeze(dim=2) * trmat_ac_prev.unsqueeze(dim=0), dim=1)
        prob_mat_prev = prob_mat_pri * prob_mat_sec
    else:  # sum over all the possible one-hot encodings
        prob_mat_prev = torch.zeros_like(prob_mat_pri)  # for accumulation
        for idx_type in range(n_types):
            prob_mat_onht = torch.zeros_like(prob_mat_pri)
            prob_mat_onht[:, idx_type] = 1.0
            prob_mat_sec = torch.sum(prob_mat_onht.unsqueeze(dim=2) * trmat_ac_prev.unsqueeze(dim=0), dim=1)
            prob_mat_tmp = prob_mat_pri * prob_mat_sec
            prob_mat_tmp /= torch.sum(prob_mat_tmp, dim=1, keepdim=True)
            prob_mat_prev += prob_mat_init[:, idx_type].unsqueeze(dim=1) * prob_mat_tmp

    # normalize probabilistic distributions
    prob_mat_prev = torch.clip(prob_mat_prev, min=0.0)
    prob_mat_prev = nn.functional.normalize(prob_mat_prev, p=1.0, dim=1)

    return prob_mat_prev


def intp_mutual_seq(aa_seq_orig, aa_seq_pert, pmsk_vec,alpha_bar_prev, alpha_bar_curr, has_noise=False):
    # calculate probabilistic distributions for $x_{t - 1}$
    theta, gamma = calc_intp_coeffs(alpha_bar_prev, alpha_bar_curr, has_noise)
    aa_seq_orig_list = list(aa_seq_orig)
    aa_seq_pert_list = list(aa_seq_pert)

    for idx in range(len(aa_seq_orig_list)):
        if pmsk_vec[idx] == 1:
            chosen_index = torch.multinomial(torch.tensor([theta, gamma]), 1).item()
            aa_seq_pert_list[idx] = aa_seq_orig_list[idx] if chosen_index == 0 else aa_seq_pert_list[idx]

    return ''.join(aa_seq_pert_list)


def intp_prob_mat_cont(prob_mat_init, prob_mat_curr, alpha_bar_prev, alpha_bar_curr, has_noise=False, normalize=True):
    """Interpolate between $x_0$ (prot_mat_init) and $x_{t}$ (prot_mat_curr) - continuous.

    Args:
    * prob_mat_init: initial probabilistic distributions ($x_{0}$) of size N x C
    * prob_mat_curr: current probabilistic distributions ($x_{t}$) of size N x C
    * alpha_bar_prev: previous accumulated alpha coefficient ($\bar{\alpha}_{t-1}$)
    * alpha_bar_curr: current accumulated alpha coefficient ($\bar{\alpha}_{t}$)
    * has_noise: (optional) whether random noise is enabled (DDPM) or not (DDIM)
    * normalize: (optional) whether to normalize output probabilistic distributions

    Returns:
    * prob_mat_prev: previous probabilistic distributions ($x_{t - 1}$) of size N x C

    Reference:
    * Song et al., Denoising Diffusion Implicit Models.
    """

    # validate input tensors
    assert prob_mat_init.ndim == 2
    assert prob_mat_init.shape == prob_mat_curr.shape
    assert alpha_bar_prev.numel() == 1
    assert alpha_bar_curr.numel() == 1

    # calculate probabilistic distributions for $x_{t - 1}$
    theta, gamma = calc_intp_coeffs(alpha_bar_prev, alpha_bar_curr, has_noise)
    prob_mat_prev = theta * prob_mat_init + gamma * prob_mat_curr
    if normalize:
        prob_mat_prev = nn.functional.normalize(
            prob_mat_prev - torch.min(prob_mat_prev, dim=1, keepdim=True)[0], p=1.0, dim=1)

    return prob_mat_prev


def calc_trsl_vec(cord_tns, cmsk_mat=None):
    """Calculate the translation vector w/ optional validness masks.

    Args:
    * cord_tns: per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: (optional) per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * trsl_vec: translation vector of size 3
    """

    # configurations
    eps = 1e-6

    # calculate the translation vector
    if cmsk_mat is None:
        trsl_vec = torch.mean(cord_tns.view(-1, 3), dim=0)
    else:
        trsl_vec = torch.sum(
            cmsk_mat.unsqueeze(dim=2) * cord_tns, dim=(0, 1)) / (torch.sum(cmsk_mat) + eps)

    return trsl_vec

def intp_trsl_mat(trsl_mat_init, trsl_mat_curr, alpha_bar_prev, alpha_bar_curr, has_noise=False):
    """Interpolate between $x_0$ (trsl_mat_init) and $x_{t}$ (trsl_mat_curr).

    Args:
    * trsl_mat_init: initial translation vectors ($x_{0}$) of size N x 3
    * trsl_mat_curr: current translation vectors ($x_{t}$) of size N x 3
    * alpha_bar_prev: previous accumulated alpha coefficient ($\bar{\alpha}_{t-1}$)
    * alpha_bar_curr: current accumulated alpha coefficient ($\bar{\alpha}_{t}$)
    * has_noise: (optional) whether random noise is enabled (DDPM) or not (DDIM)

    Returns:
    * trsl_mat_prev: previous translation vectors ($x_{t - 1}$) of size N x 3

    Reference:
    * Song et al., Denoising Diffusion Implicit Models.

    Notes:
    * For interpolation between $x_{t}$ and ground-truth $x_{0}$, <has_noise> must be set to True.
    * For interpolation between $x_{t}$ and predicted $\tilde{x}_{0}$, <has_noise> can be set to
        either True for DDPM sampling or False for DDIM sampling.
    """

    # validate input tensors
    assert trsl_mat_init.ndim == 2
    n_smpls = trsl_mat_init.shape[0]
    assert list(trsl_mat_init.shape) == [n_smpls, 3]
    assert list(trsl_mat_curr.shape) == [n_smpls, 3]
    assert alpha_bar_prev.numel() == 1
    assert alpha_bar_curr.numel() == 1

    # calculate translation vectors for $x_{t - 1}$
    theta, gamma = calc_intp_coeffs(alpha_bar_prev, alpha_bar_curr, has_noise)
    trsl_mat_prev = theta * trsl_mat_init + gamma * trsl_mat_curr

    return trsl_mat_prev


def intp_rota_tns(rota_tns_init, rota_tns_curr, alpha_bar_prev, alpha_bar_curr, has_noise=False):
    """Interpolate between $x_0$ (rota_tns_init) and $x_{t}$ (rota_tns_curr).

    Args:
    * rota_tns_init: initial rotation matrices ($x_{0}$) of size N x 3 x 3
    * rota_tns_curr: current rotation matrices ($x_{t}$) of size N x 3 x 3
    * alpha_bar_prev: previous accumulated alpha coefficient ($\bar{\alpha}_{t-1}$)
    * alpha_bar_curr: current accumulated alpha coefficient ($\bar{\alpha}_{t}$)
    * has_noise: (optional) whether random noise is enabled (DDPM) or not (DDIM)

    Returns:
    * rota_tns_prev: previous rotation matrices ($x_{t - 1}$) of size N x 3 x 3

    Reference:
    * Leach et al., Denoising Diffusion Probabilistic Models on SO(3) for Rotational Alignment.

    Notes:
    * We believe that there is a typo in Eq. (9), where $\alpha_{t-1}$ should be $\alpha_{t}$.
    * For interpolation between $x_{t}$ and ground-truth $x_{0}$, <has_noise> must be set to True.
    * For interpolation between $x_{t}$ and predicted $\tilde{x}_{0}$, <has_noise> can be set to
        either True for DDPM sampling or False for DDIM sampling.
    """

    # validate input tensors
    assert rota_tns_init.ndim == 3
    n_smpls = rota_tns_init.shape[0]
    assert list(rota_tns_init.shape) == [n_smpls, 3, 3]
    assert list(rota_tns_curr.shape) == [n_smpls, 3, 3]
    assert alpha_bar_prev.numel() == 1
    assert alpha_bar_curr.numel() == 1

    # calculate rotation matrices for $x_{t - 1}$
    theta, gamma = calc_intp_coeffs(alpha_bar_prev, alpha_bar_curr, has_noise)
    rota_tns_init_scaled = so3_scale(rota_tns_init, theta)
    rota_tns_curr_scaled = so3_scale(rota_tns_curr, gamma)
    rota_tns_prev = torch.bmm(rota_tns_init_scaled, rota_tns_curr_scaled)

    return rota_tns_prev

# def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
#     skew_mat = (r_mat - r_mat.transpose(-1, -2))
#     sk_vec = skew2vec(skew_mat)
#     s_angle = (sk_vec).norm(p=2, dim=-1) / 2
#     c_angle = (torch.einsum('...ii', r_mat) - 1) / 2
#     angle = torch.atan2(s_angle, c_angle)
#     scale = (angle / (2 * s_angle))
#     # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
#     # by definition, scale values are 0 if rotating by 0.
#     # This also breaks down if rotating by pi, fix further down
#     scale[angle == 0.0] = 0.0
#     log_r_mat = scale[..., None, None] * skew_mat

#     # Check for NaNs caused by 180deg rotations.
#     nanlocs = log_r_mat[...,0,0].isnan()
#     nanmats = r_mat[nanlocs]
#     # We need to use an alternative way of finding the logarithm for nanmats,
#     # Use eigendecomposition to discover axis of rotation.
#     # By definition, these are symmetric, so use eigh.
#     # NOTE: linalg.eig() isn't in torch 1.8,
#     #       and torch.eig() doesn't do batched matrices
#     eigval, eigvec = torch.linalg.eigh(nanmats)
#     # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
#     # this *should* just be the last column if the docs for eigh are true.
#     nan_axes = eigvec[...,-1,:]
#     nan_angle = angle[nanlocs]
#     nan_skew = vec2skew(nan_angle[...,None] * nan_axes)
#     log_r_mat[nanlocs] = nan_skew
#     return log_r_mat

def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
    orig_dtype = r_mat.dtype
    
    # ✅ 整个函数在 float32 下计算，避免 bf16 精度问题
    r_mat = r_mat.to(torch.float32)
    
    skew_mat = (r_mat - r_mat.transpose(-1, -2))
    sk_vec = skew2vec(skew_mat)
    s_angle = sk_vec.norm(p=2, dim=-1) / 2
    c_angle = (torch.einsum('...ii', r_mat) - 1) / 2
    angle = torch.atan2(s_angle, c_angle)
    
    # ✅ 用阈值而非精确零点，避免小值除法爆炸
    safe_s_angle = s_angle.clone()
    near_zero = s_angle < 1e-6
    safe_s_angle[near_zero] = 1.0  # 临时填充，防止除零
    
    scale = angle / (2 * safe_s_angle)
    scale[near_zero] = 0.0  # 旋转角≈0时 scale 定义为 0
    
    log_r_mat = scale[..., None, None] * skew_mat

    # 处理 180° 旋转导致的 NaN（此时 s_angle 不小，但 log 无法由 skew 唯一确定）
    nanlocs = log_r_mat[..., 0, 0].isnan()
    if nanlocs.any():
        nanmats = r_mat[nanlocs]  # 已经是 float32
        eigval, eigvec = torch.linalg.eigh(nanmats)
        nan_axes = eigvec[..., -1, :]
        nan_angle = angle[nanlocs]
        nan_skew = vec2skew(nan_angle[..., None] * nan_axes)
        log_r_mat[nanlocs] = nan_skew

    # ✅ 转回原始 dtype
    return log_r_mat.to(orig_dtype)

def so3_scale(rmat, scalars):
    '''Scale the magnitude of a rotation matrix,
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily. and multiply instead.
    '''
    logs = log_rmat(rmat)
    scaled_logs = logs * scalars[..., None, None]
    out = torch.matrix_exp(scaled_logs)
    return out

def rota2ypr(rota_tns):
    """Convert rotation matrices into yaw-pitch-roll angles.

    Args:
    * rota_tns: rotation matrices of size N x 3 x 3

    Returns:
    * yaw: 1st components in Euler angles of size N
    * ptc: 2nd components in Euler angles of size N
    * rll: 3rd components in Euler angles of size N

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.5.1.
    """

    # configurations
    tol = 1e-4  # determine whether a degenerate case is encountered
    eps = 1e-6

    # extract entries from rotation matrices
    r11, _, r13 = [x.squeeze(dim=1) for x in torch.split(rota_tns[:, 0], 1, dim=1)]
    r21, _, r23 = [x.squeeze(dim=1) for x in torch.split(rota_tns[:, 1], 1, dim=1)]
    r31, r32, r33 = [x.squeeze(dim=1) for x in torch.split(rota_tns[:, 2], 1, dim=1)]

    # recover pitch, yaw, and roll angles (naive implementation)
    ptc = torch.atan2(-r31, torch.sqrt(torch.square(r11) + torch.square(r21)))
    yaw = torch.where(
        torch.gt(torch.abs(torch.abs(ptc) - np.pi / 2.0), tol),
        torch.atan2(r21, r11 + eps),
        torch.where(torch.gt(ptc, 0.0), torch.atan2(r23, r13 + eps), torch.atan2(-r23, -r13 + eps)),
    )
    rll = torch.where(
        torch.gt(torch.abs(torch.abs(ptc) - np.pi / 2.0), tol),
        torch.atan2(r32, r33 + eps),
        torch.zeros_like(ptc),
    )

    return yaw, ptc, rll


def ypr2quat(yaw, ptc, rll):
    """Convert yaw-pitch-roll angles into quaternion vectors.

    Args:
    * yaw: 1st components in Euler angles of size N
    * ptc: 2nd components in Euler angles of size N
    * rll: 3rd components in Euler angles of size N

    Returns:
    * qr: 1st components in quaternion vectors of size N
    * qx: 2nd components in quaternion vectors of size N
    * qy: 3rd components in quaternion vectors of size N
    * qz: 4th components in quaternion vectors of size N

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.1.1.
    """

    # calculate normalized quaternion vectors
    ptc_s, ptc_c = torch.sin(ptc / 2.0), torch.cos(ptc / 2.0)
    yaw_s, yaw_c = torch.sin(yaw / 2.0), torch.cos(yaw / 2.0)
    rll_s, rll_c = torch.sin(rll / 2.0), torch.cos(rll / 2.0)
    qr = rll_c * ptc_c * yaw_c + rll_s * ptc_s * yaw_s
    qx = rll_s * ptc_c * yaw_c - rll_c * ptc_s * yaw_s
    qy = rll_c * ptc_s * yaw_c + rll_s * ptc_c * yaw_s
    qz = rll_c * ptc_c * yaw_s - rll_s * ptc_s * yaw_c

    return qr, qx, qy, qz

def rota2quat(rota_tns, quat_type='full'):
    """Convert rotation matrices into full / partial quaternion vectors.

    Args:
    * rota_tns: rotation matrices of size N x 3 x 3
    * quat_type: type of quaternion vectors (choices: 'full' / 'part')

    Returns:
    * quat_mat: quaternion vectors of size N x 4 (full) or N x 3 (part)
    """

    # configurations
    eps = 1e-6

    # convert rotation matrices into raw components in quaternion vectors
    yaw, ptc, rll = rota2ypr(rota_tns)
    qr, qx, qy, qz = ypr2quat(yaw, ptc, rll)

    # build quaternion vectors
    if quat_type == 'full':
        quat_mat = torch.stack([qr, qx, qy, qz], dim=1)
        quat_mat = quat_mat * torch.sign(quat_mat[:, :1] + eps)  # qr: non-negative
    elif quat_type == 'part':
        qa = qx / (qr + eps)
        qb = qy / (qr + eps)
        qc = qz / (qr + eps)
        quat_mat = torch.stack([qa, qb, qc], dim=1)
    else:
        raise ValueError(f'unrecognized type of quaternion vectors: {quat_type}')

    return quat_mat


def orthogonalise(mat):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We SVD decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    orth_mat = mat.clone()
    u, s, vh = torch.linalg.svd(mat[..., :3, :3])
    orth_mat[..., :3, :3] = u @ torch.diag_embed(s.round()) @ vh
    return orth_mat

def aa_to_rmat(rot_axis: torch.Tensor, ang: torch.Tensor):
    '''Generates a rotation matrix (3x3) from axis-angle form

        `rot_axis`: Axis to rotate around, defined as vector from origin.
        `ang`: rotation angle
        '''
    eps = 1e-6
    rot_axis_n = rot_axis / (rot_axis.norm(p=2, dim=-1, keepdim=True) + eps)
    sk_mats = vec2skew(rot_axis_n)
    log_rmats = sk_mats * ang[..., None]
    rot_mat = torch.matrix_exp(log_rmats)
    return orthogonalise(rot_mat)

def rmat_to_aa(r_mat) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Calculates axis and angle of rotation from a rotation matrix.

        returns angles in [0,pi] range.

        `r_mat`: rotation matrix.
        '''
    eps = 1e-6
    log_mat = log_rmat(r_mat)
    skew_vec = skew2vec(log_mat)
    angle = skew_vec.norm(p=2, dim=-1, keepdim=True)
    axis = skew_vec / (angle + eps)
    return axis, angle


class IsotropicGaussianSO3(Distribution):
    arg_constraints = {'eps': constraints.positive}

    def __init__(self, eps: torch.Tensor, mean: torch.Tensor = torch.eye(3)):
        self.eps = eps
        self._mean = mean.to(self.eps)
        self._mean_inv = self._mean.transpose(-1, -2)  # orthonormal so inverse = Transpose
        pdf_sample_locs = pi * torch.linspace(0, 1.0, 1000) ** 3.0  # Pack more samples near 0
        pdf_sample_locs = pdf_sample_locs.to(self.eps).unsqueeze(-1)
        # As we're sampling using axis-angle form
        # and need to account for the change in density
        # Scale by 1-cos(t)/pi for sampling
        with torch.no_grad():
            pdf_sample_vals = self._eps_ft(pdf_sample_locs) * ((1 - pdf_sample_locs.cos()) / pi)
        # Set to 0.0, otherwise there's a divide by 0 here
        pdf_sample_vals[(pdf_sample_locs == 0).expand_as(pdf_sample_vals)] = 0.0

        # Trapezoidal integration
        pdf_val_sums = pdf_sample_vals[:-1, ...] + pdf_sample_vals[1:, ...]
        pdf_loc_diffs = torch.diff(pdf_sample_locs, dim=0)
        self.trap = (pdf_loc_diffs * pdf_val_sums / 2).cumsum(dim=0)
        self.trap = self.trap/self.trap[-1,None]
        self.trap_loc = pdf_sample_locs[1:]
        super().__init__()

    def sample(self, sample_shape=torch.Size()):
        # Consider axis-angle form.
        axes = torch.randn((*sample_shape, *self.eps.shape, 3)).to(self.eps)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        # Inverse transform sampling based on numerical approximation of CDF
        unif = torch.rand((*sample_shape, *self.eps.shape), device=self.trap.device)
        idx_1 = (self.trap <= unif[None, ...]).sum(dim=0)
        idx_0 = torch.clamp(idx_1 - 1,min=0)

        trap_start = torch.gather(self.trap, 0, idx_0[..., None])[..., 0]
        trap_end = torch.gather(self.trap, 0, idx_1[..., None])[..., 0]

        trap_diff = torch.clamp((trap_end - trap_start), min=1e-6)
        weight = torch.clamp(((unif - trap_start) / trap_diff), 0, 1)
        angle_start = self.trap_loc[idx_0, 0]
        angle_end = self.trap_loc[idx_1, 0]
        angles = torch.lerp(angle_start, angle_end, weight)[..., None]
        out = self._mean @ aa_to_rmat(axes, angles)
        return out

    def sample_batch(self, sample_shape=torch.Size()):
        """Batch-mode sampling (the default routine cannot handle <sample_shape> correctly)."""

        # randomly sample rotational axes from the unit-sphere
        batch_size = sample_shape.numel()
        axes = torch.randn((batch_size, *self.eps.shape, 3)).to(self.eps)
        axes = axes / axes.norm(dim=-1, keepdim=True)

        # randomly sample rotational angles
        unif = torch.rand((batch_size, *self.eps.shape), device=self.trap.device)
        idx_1 = (self.trap.unsqueeze(dim=1) <= unif[None, ...]).sum(dim=0)
        idx_0 = torch.clamp(idx_1 - 1,min=0)
        trap_start = torch.gather(self.trap, 0, idx_0)
        trap_end = torch.gather(self.trap, 0, idx_1)
        trap_diff = torch.clamp((trap_end - trap_start), min=1e-6)
        weight = torch.clamp(((unif - trap_start) / trap_diff), 0, 1)
        angle_start = self.trap_loc[idx_0, 0]
        angle_end = self.trap_loc[idx_1, 0]
        angles = torch.lerp(angle_start, angle_end, weight)[..., None]

        # convert axis-angle representation into rotational matrices
        out = self._mean @ aa_to_rmat(axes.view(-1, 3), angles.view(-1, 1))
        out = out.view(*sample_shape, *self.eps.shape, 3, 3)

        return out

    def _eps_ft(self, t: torch.Tensor) -> torch.Tensor:
        var_d = self.eps.double()**2
        t_d = t.double()
        vals = sqrt(pi) * var_d ** (-3 / 2) * torch.exp(var_d / 4) * torch.exp(-((t_d / 2) ** 2) / var_d) \
               * (t_d - torch.exp((-pi ** 2) / var_d)
                  * ((t_d - 2 * pi) * torch.exp(pi * t_d / var_d) + (
                            t_d + 2 * pi) * torch.exp(-pi * t_d / var_d))
                  ) / (2 * torch.sin(t_d / 2))
        vals[vals.isinf()] = 0.0
        vals[vals.isnan()] = 0.0

        # using the value of the limit t -> 0 to fix nans at 0
        t_big, _ = torch.broadcast_tensors(t_d, var_d)
        # Just trust me on this...
        # This doesn't fix all nans as a lot are still too big to flit in float32 here
        vals[t_big == 0] = sqrt(pi) * (var_d * torch.exp(2 * pi ** 2 / var_d)
                                       - 2 * var_d * torch.exp(pi ** 2 / var_d)
                                       + 4 * pi ** 2 * var_d * torch.exp(pi ** 2 / var_d)
                                       ) * torch.exp(var_d / 4 - (2 * pi ** 2) / var_d) / var_d ** (5 / 2)
        return vals.float()

    def log_prob(self, rotations):
        _, angles = rmat_to_aa(rotations)
        probs = self._eps_ft(angles)
        return probs.log()

    @property
    def mean(self):
        return self._mean


class OnlineIGSO3Schedule:
    """Online IGSO(3) sampler parameterized by actual rotation-angle RMS."""

    def __init__(self, angle_rms: torch.Tensor, seed: int = 0, calibration_size: int = 512):
        angle_rms = torch.as_tensor(angle_rms, dtype=torch.float64, device="cpu").reshape(-1)
        if angle_rms.numel() < 2 or angle_rms[0] != 0:
            raise ValueError("angle_rms must contain t=0 identity followed by at least one noisy step")
        if not torch.isfinite(angle_rms).all() or torch.any(angle_rms[1:] <= 0):
            raise ValueError("non-zero timestep angle RMS values must be finite and positive")
        if torch.any(angle_rms[1:] < angle_rms[:-1]):
            raise ValueError("angle RMS schedule must be monotonically non-decreasing")

        self.angle_rms = angle_rms
        self.n_steps = angle_rms.numel() - 1
        self.haar_rms = float(sqrt(pi ** 2 / 3.0 + 2.0))
        if abs(float(angle_rms[-1]) - self.haar_rms) > 5e-4:
            raise ValueError("the terminal angle RMS must equal the Haar-uniform SO(3) RMS")

        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.eps = torch.zeros_like(angle_rms)
        self.angle_locs = torch.empty(0, dtype=torch.float64)
        self.angle_cdf = torch.empty((0, max(self.n_steps - 1, 0)), dtype=torch.float64)

        if self.n_steps > 1:
            n_log = calibration_size // 3
            eps_grid = torch.cat([
                torch.logspace(math.log10(0.003), math.log10(0.1), n_log, dtype=torch.float64),
                torch.linspace(0.1, 3.0, calibration_size - n_log + 1, dtype=torch.float64)[1:],
            ])
            calibration = IsotropicGaussianSO3(eps_grid)
            rms_grid = self._cdf_angle_rms(calibration.trap, calibration.trap_loc[:, 0])
            rms_grid = torch.cummax(rms_grid, dim=0).values
            targets = angle_rms[1:-1]
            if targets.numel() and (targets[0] < rms_grid[0] or targets[-1] > rms_grid[-1]):
                raise ValueError(
                    f"angle RMS target range [{float(targets[0]):.6f}, {float(targets[-1]):.6f}] "
                    f"is outside IGSO3 calibration range [{float(rms_grid[0]):.6f}, {float(rms_grid[-1]):.6f}]"
                )
            self.eps[1:-1] = self._interpolate_eps(targets, rms_grid, eps_grid)
            distributions = IsotropicGaussianSO3(self.eps[1:-1])
            self.angle_locs = distributions.trap_loc[:, 0].contiguous()
            self.angle_cdf = distributions.trap.transpose(0, 1).contiguous()

    @staticmethod
    def _cdf_angle_rms(cdf: torch.Tensor, upper_locs: torch.Tensor) -> torch.Tensor:
        zero_cdf = torch.zeros((1, cdf.shape[1]), dtype=cdf.dtype, device=cdf.device)
        mass = torch.diff(torch.cat([zero_cdf, cdf], dim=0), dim=0).clamp_min(0.0)
        lower_locs = torch.cat([torch.zeros(1, dtype=upper_locs.dtype), upper_locs[:-1]])
        midpoint_sq = ((lower_locs + upper_locs) * 0.5).square().unsqueeze(-1)
        return torch.sqrt((mass * midpoint_sq).sum(0) / mass.sum(0).clamp_min(1e-12))

    @staticmethod
    def _interpolate_eps(target: torch.Tensor, rms_grid: torch.Tensor, eps_grid: torch.Tensor) -> torch.Tensor:
        if target.numel() == 0:
            return target.clone()
        idx_hi = torch.searchsorted(rms_grid, target).clamp(1, rms_grid.numel() - 1)
        idx_lo = idx_hi - 1
        rms_lo, rms_hi = rms_grid[idx_lo], rms_grid[idx_hi]
        weight = (target - rms_lo) / (rms_hi - rms_lo).clamp_min(1e-12)
        return torch.lerp(eps_grid[idx_lo], eps_grid[idx_hi], weight)

    @staticmethod
    def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        x, y, z = axis
        zero = torch.zeros((), dtype=axis.dtype)
        skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
        eye = torch.eye(3, dtype=axis.dtype)
        return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)

    def _sample_igso3(self, idx_step: int) -> torch.Tensor:
        cdf = self.angle_cdf[idx_step - 1]
        uniform = torch.rand((), generator=self.generator, dtype=torch.float64)
        idx_hi = int(torch.searchsorted(cdf, uniform).clamp(max=cdf.numel() - 1).item())
        cdf_lo = cdf.new_tensor(0.0) if idx_hi == 0 else cdf[idx_hi - 1]
        cdf_hi = cdf[idx_hi]
        angle_lo = self.angle_locs.new_tensor(0.0) if idx_hi == 0 else self.angle_locs[idx_hi - 1]
        angle_hi = self.angle_locs[idx_hi]
        weight = ((uniform - cdf_lo) / (cdf_hi - cdf_lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        angle = torch.lerp(angle_lo, angle_hi, weight)
        axis = torch.randn(3, generator=self.generator, dtype=torch.float64)
        axis = axis / axis.norm().clamp_min(1e-12)
        return self._axis_angle_to_matrix(axis, angle)

    def _sample_haar(self) -> torch.Tensor:
        quat = torch.randn(4, generator=self.generator, dtype=torch.float64)
        quat = quat / quat.norm().clamp_min(1e-12)
        w, x, y, z = quat
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ]).reshape(3, 3)

    def sample(self, idx_step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx_step = int(idx_step)
        if idx_step < 0 or idx_step > self.n_steps:
            raise ValueError(f"rotation timestep {idx_step} is outside [0, {self.n_steps}]")
        if idx_step == 0:
            rotation = torch.eye(3, dtype=torch.float64)
        elif idx_step == self.n_steps:
            rotation = self._sample_haar()
        else:
            rotation = self._sample_igso3(idx_step)
        rotation = rotation.to(device=device, dtype=dtype)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            rotation_f32 = rotation.float()
            eye = torch.eye(3, device=device, dtype=torch.float32)
            orth_error = (rotation_f32.transpose(-1, -2) @ rotation_f32 - eye).abs().amax()
            det_error = (torch.det(rotation_f32) - 1.0).abs()
            is_finite = torch.isfinite(rotation_f32).all()
        if not is_finite or orth_error > 5e-4 or det_error > 5e-4:
            raise RuntimeError(
                f"invalid online SO(3) sample at t={idx_step}: "
                f"orth_error={float(orth_error):.3e}, det_error={float(det_error):.3e}"
            )
        return rotation


class IGSO3Buffer():
    """Buffered rotational matrices sampled from IGSO(3) distributions."""

    def __init__(self, sigma_min=0.0001, sigma_max=1.0, reso=0.0001, buf_size=1024):
        """Constructor function."""

        # setup configurations
        self.sigmas = np.arange(sigma_min, sigma_max, reso).tolist()
        self.buf_size = buf_size  # number of buffered rotational matrices per distribution

        # addditional configurations
        self.buf_list = []  # will be initialized via build() or load()


    def build(self):
        """Build buffers of rotational matrices, one per IGSO(3) distribution."""

        # check the early-exit condition
        if len(self.buf_list) == len(self.sigmas):
            return

        # build buffers of rotational matrices
        self.buf_list = []
        for sigma in self.sigmas:
            igso3 = IsotropicGaussianSO3(eps=torch.tensor([sigma]))
            rota_tns = igso3.sample_batch(torch.Size([self.buf_size]))[:, 0]
            self.buf_list.append(rota_tns)


    def save(self, path):
        """Save buffered rotational matrices to file."""

        snapshot = {
            'sigmas': self.sigmas,
            'buf_size': self.buf_size,
            'buf_list': self.buf_list,
        }
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        torch.save(snapshot, path)


    def load(self, path):
        """Load buffered rotational matrices from file."""

        snapshot = torch.load(path, map_location='cpu',weights_only=False)
        self.sigmas = snapshot['sigmas']
        self.buf_size = snapshot['buf_size']
        self.buf_list = snapshot['buf_list']
        assert len(self.buf_list) == len(self.sigmas)
        assert all(x.shape[0] == self.buf_size for x in self.buf_list)


    def sample(self, sigma, batch_size):
        """Sample a mini-batch of rotational matrices from the closest IGSO(3) distribution."""

        # find the closest IGSO(3) distribution
        assert isinstance(sigma, float), 'float data type is expected for higher efficiency'
        idx_opt = np.argmin([abs(sigma - x) for x in self.sigmas])
        rota_tns_buf = self.buf_list[idx_opt]

        # sample a mini-batch of rotational matrices
        idxs_smpl = random.choices(range(self.buf_size), k=batch_size)  # w/ replacement
        rota_tns_out = rota_tns_buf[idxs_smpl]

        return rota_tns_out

def replace_with_mask(original_string, new_string, mask):
    """Replace the characters in the original string with the new string based on the mask.

    Args:
    * original_string: original string
    * new_string: new string
    * mask: mask of booleans

    Returns:
    * replaced_string: replaced string
    """
    original_list = list(original_string)
    new_list = list(new_string)
    for i, should_replace in enumerate(mask):
        if should_replace:
            original_list[i] = new_list[i]
    return ''.join(original_list)


def so3_log_vector(rotation: torch.Tensor) -> torch.Tensor:
    rotation = rotation.float()
    skew = 0.5 * torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sin_angle = torch.linalg.norm(skew, dim=-1)
    cos_angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.atan2(sin_angle, cos_angle)
    rotvec = skew * (angle / sin_angle.clamp_min(1e-7)).unsqueeze(-1)
    rotvec = torch.where((angle < 1e-5).unsqueeze(-1), skew, rotvec)

    near_pi = (torch.pi - angle).abs() < 1e-4
    if near_pi.any():
        sym = 0.5 * (
            rotation[near_pi]
            + torch.eye(3, device=rotation.device, dtype=torch.float32)
        )
        _, eigenvectors = torch.linalg.eigh(sym)
        axes = eigenvectors[..., -1]
        max_indices = axes.abs().argmax(dim=-1, keepdim=True)
        signs = torch.gather(axes, -1, max_indices).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        rotvec[near_pi] = axes * signs * angle[near_pi].unsqueeze(-1)
    return rotvec