from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class CrossPairBiasedAttention(nn.Module):
    def __init__(
        self, c_q: int, c_kv: int | None = None, n_heads: int = 8, dropout: float = 0.0
    ) -> None:
        super().__init__()
        c_kv = c_q if c_kv is None else c_kv
        if c_q % n_heads != 0:
            raise ValueError(f"c_q={c_q} must be divisible by n_heads={n_heads}")

        self.c_q, self.n_heads = c_q, n_heads
        self.head_dim = c_q // n_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.q_proj = nn.Linear(c_q, c_q, bias=False)
        self.k_proj = nn.Linear(c_kv, c_q, bias=False)
        self.v_proj = nn.Linear(c_kv, c_q, bias=False)
        self.out_proj = nn.Linear(c_q, c_q)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, *, 
        pair_bias: torch.Tensor | None = None, query_mask: torch.Tensor | None = None, 
        key_mask: torch.Tensor | None = None, pair_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, n_query, _ = query.shape
        n_key = key_value.shape[1]

        # Project and reshape to (bsz, n_heads, seq_len, head_dim)
        q = self.q_proj(query).view(bsz, n_query, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(bsz, n_key, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(bsz, n_key, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        if pair_bias is not None:
            if pair_bias.shape != (bsz, self.n_heads, n_query, n_key):
                raise ValueError(f"Unexpected pair bias shape: {tuple(pair_bias.shape)}")
            scores = scores + pair_bias.to(scores.dtype)

        # Combine all masks into a single allowed boolean tensor
        allowed = torch.ones((bsz, n_query, n_key), dtype=torch.bool, device=query.device)
        if query_mask is not None: allowed &= query_mask.to(torch.bool).unsqueeze(-1)
        if key_mask is not None: allowed &= key_mask.to(torch.bool).unsqueeze(1)
        if pair_mask is not None: allowed &= pair_mask.to(torch.bool)

        allowed_h = allowed.unsqueeze(1)
        masked_scores = scores.float().masked_fill(~allowed_h, -1.0e9)

        # Safe softmax
        score_max = masked_scores.amax(dim=-1, keepdim=True)
        attention = torch.exp(masked_scores - score_max) * allowed_h.to(masked_scores.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        attention = self.dropout(attention.to(v.dtype))

        # Compute output and project back
        output = torch.einsum("bhqk,bhkd->bhqd", attention, v)
        output = output.transpose(1, 2).contiguous().view(bsz, n_query, self.c_q)
        output = self.out_proj(output)

        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)

        return output


class CDRLoopHead(nn.Module):
    def __init__(
        self, c_s: int = 384, c_z: int = 128, c_e: int = 64, n_loop_types: int = 6, 
        max_positions: int = 64, c_token: int = 384, c_atom: int = 192, n_atom: int = 14, 
        n_heads: int = 8, rbf_bins: int = 32, rbf_max_distance: float = 32.0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.n_atom = n_atom
        self.n_heads = n_heads

        # Embeddings
        self.loop_type = nn.Embedding(n_loop_types, 32)
        self.local_pos = nn.Embedding(max_positions, 32)
        
        # Encoders and Trunks
        self.single_cond = nn.Sequential(
            nn.LayerNorm(c_s + c_e + 32 + 32), nn.Linear(c_s + c_e + 32 + 32, c_token),
            nn.SiLU(), nn.Linear(c_token, c_token)
        )
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(n_atom * 3 + c_token + n_atom), nn.Linear(n_atom * 3 + c_token + n_atom, c_atom),
            nn.SiLU(), nn.Linear(c_atom, c_atom)
        )
        self.token_trunk = nn.Sequential(
            nn.LayerNorm(c_token + c_atom), nn.Linear(c_token + c_atom, c_token),
            nn.SiLU(), nn.Linear(c_token, c_token)
        )

        # Intra-loop Attention
        self.intra_loop_attention = CrossPairBiasedAttention(c_q=c_token, n_heads=n_heads, dropout=dropout)
        self.intra_loop_norm = nn.LayerNorm(c_token)
        self.intra_loop_ffn = nn.Sequential(
            nn.LayerNorm(c_token), nn.Linear(c_token, c_token * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(c_token * 2, c_token)
        )

        # Cross-loop Attention
        self.cross_loop_attention = CrossPairBiasedAttention(c_q=c_token, n_heads=n_heads, dropout=dropout)
        self.cross_loop_norm = nn.LayerNorm(c_token)
        self.cross_loop_ffn = nn.Sequential(
            nn.LayerNorm(c_token), nn.Linear(c_token, c_token * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(c_token * 2, c_token)
        )

        # Antigen Cross-Attention
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(c_s), nn.Linear(c_s, c_token), nn.SiLU(), nn.Linear(c_token, c_token)
        )
        self.antigen_attention = CrossPairBiasedAttention(c_q=c_token, c_kv=c_token, n_heads=n_heads, dropout=dropout)
        self.antigen_norm = nn.LayerNorm(c_token)
        self.antigen_ffn = nn.Sequential(
            nn.LayerNorm(c_token), nn.Linear(c_token, c_token * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(c_token * 2, c_token)
        )

        # Pair biases and Distance (RBF) biases
        self.cross_loop_pair_norm = nn.LayerNorm(c_z)
        self.loop_antigen_pair_norm = nn.LayerNorm(c_z)
        self.cross_loop_pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.loop_antigen_pair_bias = nn.Linear(c_z, n_heads, bias=False)
        self.cross_loop_distance_bias = nn.Linear(rbf_bins, n_heads, bias=False)
        self.loop_antigen_distance_bias = nn.Linear(rbf_bins, n_heads, bias=False)

        self.register_buffer("rbf_centers", torch.linspace(0.0, float(rbf_max_distance), rbf_bins), persistent=False)
        self.rbf_width = float(rbf_max_distance) / max(rbf_bins - 1, 1)

        # Decoders & Heads
        self.atom_decoder = nn.Sequential(
            nn.LayerNorm(c_atom + c_token + n_atom * 3), nn.Linear(c_atom + c_token + n_atom * 3, c_atom),
            nn.SiLU(), nn.Linear(c_atom, c_atom), nn.SiLU()
        )
        self.coord_head = nn.Linear(c_atom, n_atom * 3)
        self.occ_head = nn.Linear(c_token, 1)
        self.topology_head = nn.Sequential(nn.LayerNorm(c_atom), nn.Linear(c_atom, n_atom * 3))

        # Sequence head (co-design, scheme B): predict 20-class residue-type
        # logits directly from the per-residue loop token, decoupling type from
        # the geometry readout. Trained with cross-entropy (default weight 0).
        self.n_aa_types = 20
        self.seq_head = nn.Sequential(
            nn.LayerNorm(c_token), nn.Linear(c_token, c_token),
            nn.SiLU(), nn.Linear(c_token, self.n_aa_types),
        )

        self.noise_embed = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.noise_to_token = nn.Linear(32, c_token)

        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

    def _distance_rbf(self, distance: torch.Tensor) -> torch.Tensor:
        centers = self.rbf_centers.to(device=distance.device, dtype=distance.dtype)
        width = max(self.rbf_width, 1.0e-6)
        return torch.exp(-((distance.unsqueeze(-1) - centers) / width) ** 2)

    def _build_pair_bias(
        self, pair_features: torch.Tensor, distances: torch.Tensor, pair_norm: nn.LayerNorm,
        pair_projection: nn.Linear, distance_projection: nn.Linear
    ) -> torch.Tensor:
        pair_component = pair_projection(pair_norm(pair_features))
        distance_component = distance_projection(self._distance_rbf(distances))
        return (pair_component + distance_component).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _expand_sigma(sigma: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = sigma.to(device=device, dtype=dtype).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        elif sigma.numel() != batch_size:
            raise ValueError(f"Unexpected sigma shape: {tuple(sigma.shape)}")
        return sigma

    def forward(
        self, loop_sfea: torch.Tensor, loop_encd: torch.Tensor, loop_xt_scaled: torch.Tensor,
        loop_type_ids: torch.Tensor, local_position_ids: torch.Tensor, loop_valid_res_mask: torch.Tensor,
        loop_atom_valid_mask: torch.Tensor, cdr_sigma: torch.Tensor, full_sfea: torch.Tensor,
        antigen_mask: torch.Tensor, cross_loop_pair_features: torch.Tensor, cross_loop_distances: torch.Tensor,
        cross_loop_pair_mask: torch.Tensor, loop_antigen_pair_features: torch.Tensor, loop_antigen_distances: torch.Tensor,
    ) -> dict:
        bsz, n_loop, lmax = loop_xt_scaled.shape[:3]
        device, dtype = loop_sfea.device, loop_sfea.dtype
        valid_res_mask = loop_valid_res_mask.to(torch.bool)
        atom_mask = loop_atom_valid_mask.to(dtype)

        # 1. Feature initialization & Noise condition embedding
        type_feat = self.loop_type(loop_type_ids.to(device)).view(bsz, n_loop, 1, -1).expand(-1, -1, lmax, -1)
        pos_feat = self.local_pos(local_position_ids.to(device)).view(1, 1, lmax, -1).expand(bsz, n_loop, -1, -1)
        sigma = self._expand_sigma(cdr_sigma, bsz, device, dtype).clamp_min(1.0e-8)

        c_noise = 0.25 * torch.log(sigma)
        noise_feat = self.noise_embed(c_noise.unsqueeze(-1)).view(bsz, 1, 1, -1).expand(-1, n_loop, lmax, -1)

        token_cond = self.single_cond(torch.cat([loop_sfea, loop_encd, type_feat, pos_feat], dim=-1))
        token_cond = (token_cond + self.noise_to_token(noise_feat)) * valid_res_mask.unsqueeze(-1).to(dtype)

        # 2. Atom & Token Encoders
        xt_in = loop_xt_scaled.to(dtype).reshape(bsz, n_loop, lmax, -1)
        atom_encoder_in = torch.cat([xt_in, token_cond, atom_mask], dim=-1)
        atom_feat = self.atom_encoder(atom_encoder_in) * valid_res_mask.unsqueeze(-1).to(dtype)

        token_feat = self.token_trunk(torch.cat([token_cond, atom_feat], dim=-1))
        token_feat = token_feat * valid_res_mask.unsqueeze(-1).to(dtype)

        # 3. Intra-loop Attention
        intra_tokens = token_feat.reshape(bsz * n_loop, lmax, -1)
        intra_mask = valid_res_mask.reshape(bsz * n_loop, lmax)
        
        intra_update = self.intra_loop_attention(intra_tokens, intra_tokens, query_mask=intra_mask, key_mask=intra_mask)
        intra_tokens = self.intra_loop_norm(intra_tokens + intra_update)
        intra_tokens = intra_tokens + self.intra_loop_ffn(intra_tokens)
        intra_tokens = intra_tokens * intra_mask.unsqueeze(-1).to(intra_tokens.dtype)

        # 4. Cross-loop Attention
        token_flat = intra_tokens.view(bsz, n_loop * lmax, -1)
        valid_flat = valid_res_mask.reshape(bsz, n_loop * lmax)
        cross_loop_bias = self._build_pair_bias(
            cross_loop_pair_features, cross_loop_distances, self.cross_loop_pair_norm, 
            self.cross_loop_pair_bias, self.cross_loop_distance_bias
        )

        cross_loop_update = self.cross_loop_attention(
            token_flat, token_flat, pair_bias=cross_loop_bias, query_mask=valid_flat, 
            key_mask=valid_flat, pair_mask=cross_loop_pair_mask
        )
        token_flat = self.cross_loop_norm(token_flat + cross_loop_update)
        token_flat = token_flat + self.cross_loop_ffn(token_flat)
        token_flat = token_flat * valid_flat.unsqueeze(-1).to(token_flat.dtype)

        # 5. Antigen Cross-Attention
        antigen_tokens = self.antigen_projection(full_sfea)
        antigen_mask = antigen_mask.to(torch.bool)
        loop_antigen_bias = self._build_pair_bias(
            loop_antigen_pair_features, loop_antigen_distances, self.loop_antigen_pair_norm, 
            self.loop_antigen_pair_bias, self.loop_antigen_distance_bias
        )

        antigen_update = self.antigen_attention(
            token_flat, antigen_tokens, pair_bias=loop_antigen_bias, query_mask=valid_flat, key_mask=antigen_mask
        )
        token_flat = self.antigen_norm(token_flat + antigen_update)
        token_flat = token_flat + self.antigen_ffn(token_flat)
        token_flat = token_flat * valid_flat.unsqueeze(-1).to(token_flat.dtype)

        # 6. Decoders: Outputs formulation
        token_feat = token_flat.view(bsz, n_loop, lmax, -1)
        atom_decoder_in = torch.cat([atom_feat, token_feat, xt_in], dim=-1)
        atom_hidden = self.atom_decoder(atom_decoder_in)

        x0_norm = self.coord_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        x0_norm = x0_norm * loop_atom_valid_mask.unsqueeze(-1).to(x0_norm.dtype)

        # Predict topology and occupancy logits
        pi_logits = self.topology_head(atom_hidden).view(bsz, n_loop, lmax, self.n_atom, 3)
        occ_raw = self.occ_head(token_feat).squeeze(-1)
        
        # Cumulative flip trick for auto-regressive or masking proxy
        occ_logits = torch.flip(torch.cumsum(torch.flip(occ_raw, dims=[-1]), dim=-1), dims=[-1])
        occ_logits = occ_logits.masked_fill(~valid_res_mask, -20.0)

        # Sequence-type logits per loop residue: [bsz, n_loop, lmax, 20].
        seq_logits = self.seq_head(token_feat)
        seq_logits = seq_logits * valid_res_mask.unsqueeze(-1).to(seq_logits.dtype)

        return {
            "x0_norm": x0_norm,
            "pred_occupancy_logits": occ_logits,
            "loop_update_feat": token_feat * valid_res_mask.unsqueeze(-1).to(token_feat.dtype),
            "pi_logits": pi_logits,
            "seq_logits": seq_logits,
        }
