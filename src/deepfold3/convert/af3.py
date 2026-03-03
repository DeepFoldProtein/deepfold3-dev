"""Convert AF3 Haiku parameters → PyTorch state_dict.

Strategy: explicit scope/name → PyTorch key mapping.

Also provides `extract_fourier_weights()` to read the fixed Fourier
embedding constants from the AF3 reference source code.

Weight storage conventions (Haiku, inferred from reading
references/alphafold3/src/alphafold3/model/params.py and network modules):

    hm.Linear(use_bias=False):              'weights' only
    hm.Linear(use_bias=True):               'weights' + 'bias'
    hm.LayerNorm():                         'scale' + 'offset'
    Linear(transpose_weights=True):         stores [*out_shape, in_features]
    Linear(transpose_weights=False/default):stores [in_features, *out_shape]
    hk.get_parameter('output_w', ...):      stored as 'output_w' in scope
    hk.get_parameter('output_b', ...):      stored as 'output_b' in scope

Layer stacking (hk.experimental.layer_stack):
    __layer_stack_no_per_layer   → first dim of every tensor = num_layers
    __layer_stack_with_per_layer → same

PyTorch convention:
    nn.Linear:   weight [out, in], bias [out]
    nn.LayerNorm: weight [C], bias [C]

Weight conversion:
    regular:   Haiku [in, *out] → PyTorch [prod(out), in]  → arr.reshape(in,-1).T
    transposed:Haiku [*out, in] → PyTorch [prod(out), in]  → arr.reshape(-1, in)

transpose_weights=True is used in GridSelfAttention ONLY for:
    q_projection, k_projection, gating_query
All other linears are regular (transpose_weights=False).
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import numpy as np
import torch


def _w(arr: np.ndarray, transposed: bool = False) -> torch.Tensor:
    """Haiku weight → PyTorch weight [prod(out), in]."""
    if transposed:
        in_dim = arr.shape[-1]
        out = arr.reshape(-1, in_dim)
    else:
        in_dim = arr.shape[0]
        out = arr.reshape(in_dim, -1).T
    return torch.from_numpy(out.copy()).float()


def _b(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.reshape(-1).copy()).float()


def _linear(params: dict, pt_prefix: str, transposed: bool = False):
    """Yield (key, tensor) for a Haiku Linear."""
    if "weights" in params:
        yield f"{pt_prefix}.weight", _w(params["weights"], transposed)
    if "bias" in params:
        yield f"{pt_prefix}.bias", _b(params["bias"])


def _linear_scalar(params: dict, pt_prefix: str):
    """Yield (key, tensor) for a Haiku Linear with num_input_dims=0.

    AF3 stores these weights as shape (num_output,) — a point-wise scale.
    We map to a LinearNoBias(1, num_output) with weight [num_output, 1].
    """
    if "weights" in params:
        w = params["weights"]  # shape (num_output,)
        t = torch.from_numpy(w.reshape(-1, 1).copy()).float()  # [num_output, 1]
        yield f"{pt_prefix}.weight", t


def _layernorm(params: dict, pt_prefix: str):
    """Yield (key, tensor) for a Haiku LayerNorm (scale=weight, offset=bias)."""
    if "scale" in params:
        yield f"{pt_prefix}.weight", _b(params["scale"])
    if "offset" in params:
        yield f"{pt_prefix}.bias", _b(params["offset"])


def _scale_param(params: dict, pt_key: str):
    """Yield (key, tensor) for a scale-only LayerNorm stored as a bare Parameter.

    Used for Haiku LayerNorm(create_scale=True, create_offset=False).
    The scale is stored in a nn.Parameter, not in a LayerNorm submodule.
    """
    if "scale" in params:
        yield pt_key, _b(params["scale"])


def _trimul(p: dict, prefix: str):
    """TriangleMulOutgoing / TriangleMulIncoming."""
    yield from _layernorm(p["left_norm_input"], f"{prefix}.norm_left")
    yield from _linear(p["projection"], f"{prefix}.proj")
    yield from _linear(p["gate"], f"{prefix}.gate")
    yield from _layernorm(p["center_norm"], f"{prefix}.norm_center")
    yield from _linear(p["output_projection"], f"{prefix}.out_proj")
    yield from _linear(p["gating_linear"], f"{prefix}.out_gate")


def _outer_product(scope_self: dict, sub: dict, prefix: str):
    """OuterProductMean.
    scope_self: params dict of the outer_product_mean scope itself (output_w, output_b)
    sub: sub-scope dicts keyed by sub-name.
    """
    yield from _layernorm(sub["layer_norm_input"], f"{prefix}.norm")
    yield from _linear(sub["left_projection"], f"{prefix}.proj_a")
    yield from _linear(sub["right_projection"], f"{prefix}.proj_b")
    if "output_w" in scope_self:
        yield f"{prefix}.out_w", torch.from_numpy(scope_self["output_w"].copy()).float()
    if "output_b" in scope_self:
        yield f"{prefix}.out_b", torch.from_numpy(scope_self["output_b"].copy()).float()


def _grid_attn(p: dict, prefix: str):
    """GridSelfAttention.
    q, k, gating_query use transpose_weights=True in GridSelfAttention.
    """
    yield from _layernorm(p["act_norm"], f"{prefix}.norm")
    yield from _linear(p["pair_bias_projection"], f"{prefix}.pair_bias")
    yield from _linear(p["q_projection"], f"{prefix}.to_q", transposed=True)
    yield from _linear(p["k_projection"], f"{prefix}.to_k", transposed=True)
    yield from _linear(p["v_projection"], f"{prefix}.to_v", transposed=False)
    yield from _linear(p["gating_query"], f"{prefix}.to_g", transposed=True)
    yield from _linear(p["output_projection"], f"{prefix}.out_proj")


def _msa_attn(p: dict, prefix: str):
    """MSAAttention — no transpose_weights anywhere."""
    yield from _layernorm(p["act_norm"], f"{prefix}.norm")
    yield from _layernorm(p["pair_norm"], f"{prefix}.pair_norm")
    yield from _linear(p["pair_logits"], f"{prefix}.pair_logits")
    yield from _linear(p["v_projection"], f"{prefix}.to_v")
    yield from _linear(p["gating_query"], f"{prefix}.to_g")
    yield from _linear(p["output_projection"], f"{prefix}.out_proj")


def _transition(p: dict, prefix: str):
    """TransitionBlock (input_layer_norm + transition1 + transition2)."""
    yield from _layernorm(p["input_layer_norm"], f"{prefix}.norm")
    yield from _linear(p["transition1"], f"{prefix}.linear1")
    yield from _linear(p["transition2"], f"{prefix}.linear2")


def _single_attn_flat(p: dict, prefix: str):
    """SingleAttention stored flat in PairFormer scope (keys prefixed 'single_attention_').

    In PairFormerIteration the call is:
        self_attention(single_act, seq_mask, pair_logits, config, global_config, name='single_attention_')
    with single_cond=None → adaptive_layernorm uses regular LayerNorm (scale+offset).
    adaptive_zero_init with single_cond=None → plain Linear (no gate).
    """
    yield from _layernorm(p["layer_norm"], f"{prefix}.norm")
    yield from _linear(p["q_projection"], f"{prefix}.to_q")  # use_bias=True
    yield from _linear(p["k_projection"], f"{prefix}.to_k")
    yield from _linear(p["v_projection"], f"{prefix}.to_v")
    yield from _linear(p["gating_query"], f"{prefix}.to_g")
    yield from _linear(p["transition2"], f"{prefix}.out_proj")


def _atom_cross_att_transformer(
    af3_params: dict[str, dict[str, np.ndarray]],
    scope_outer: str,  # e.g. "diffuser/~/diffusion_head/diffusion_atom_transformer_encoder/"
    pt_prefix: str,  # e.g. "diffusion_head.atom_encoder.atom_transformer"
    mark_fn=None,  # optional callable to mark scopes as mapped
) -> list[tuple[str, torch.Tensor]]:
    """Convert a CrossAttTransformer (stacked with __layer_stack_with_per_layer).

    Returns list of (pt_key, tensor).
    """
    results = []
    inner_prefix = scope_outer + "__layer_stack_with_per_layer/"

    # Determine the module name prefix embedded in the param scope names.
    # E.g. 'diffusion_atom_transformer_encoderq_projection' → mod='diffusion_atom_transformer_encoder'
    # Use q_projection to detect the module name prefix (not prefixed with ffw_).
    mod_name = ""
    for scope in af3_params:
        if scope.startswith(inner_prefix):
            suffix = scope[len(inner_prefix) :]
            if suffix.endswith("q_projection"):
                mod_name = suffix[: -len("q_projection")]
                break

    def g(param_name: str, layer_idx: int) -> dict:
        scope = inner_prefix + mod_name + param_name
        if mark_fn is not None:
            mark_fn(scope)
        raw = af3_params.get(scope, {})
        return {k: v[layer_idx] for k, v in raw.items()}

    # Number of blocks from q_projection shape
    q_scope = inner_prefix + mod_name + "q_projection"
    raw_q = af3_params.get(q_scope, {})
    if not raw_q or "weights" not in raw_q:
        return results
    n_blocks = raw_q["weights"].shape[0]

    # Pair norm + pair logits projection (outer scope, not stacked)
    pair_norm_scope = scope_outer + "pair_input_layer_norm"
    pair_proj_scope = scope_outer + "pair_logits_projection"
    if mark_fn is not None:
        mark_fn(pair_norm_scope)
        mark_fn(pair_proj_scope)
    for k, v in _scale_param(af3_params.get(pair_norm_scope, {}), f"{pt_prefix}.pair_norm_w"):
        results.append((k, v))
    for k, v in _linear(af3_params.get(pair_proj_scope, {}), f"{pt_prefix}.pair_logits_proj"):
        results.append((k, v))

    for i in range(n_blocks):
        lp = f"{pt_prefix}"  # all i blocks share the same weights (stacked), we unstack to layers[i]
        lp_a = f"{pt_prefix}.attn_blocks.{i}"
        lp_f = f"{pt_prefix}.ffn_blocks.{i}"

        # AdaLN for attention (q-side conditioning)
        # q-side AdaLN conditioning
        qln = g("qsingle_cond_layer_norm", i)
        if "scale" in qln:
            results.append((f"{lp}.attn_cond_w.{i}", _b(qln["scale"])))
        for k, v in _linear(g("qsingle_cond_scale", i), f"{lp}.attn_cond_scale.{i}"):
            results.append((k, v))
        for k, v in _linear(g("qsingle_cond_bias", i), f"{lp}.attn_cond_bias.{i}"):
            results.append((k, v))
        # k-side AdaLN conditioning
        kln = g("ksingle_cond_layer_norm", i)
        if "scale" in kln:
            results.append((f"{lp}.k_cond_w.{i}", _b(kln["scale"])))
        for k, v in _linear(g("ksingle_cond_scale", i), f"{lp}.k_cond_scale.{i}"):
            results.append((k, v))
        for k, v in _linear(g("ksingle_cond_bias", i), f"{lp}.k_cond_bias.{i}"):
            results.append((k, v))

        # Attention
        for k, v in _linear(g("q_projection", i), f"{lp_a}.to_q"):
            results.append((k, v))
        for k, v in _linear(g("k_projection", i), f"{lp_a}.to_k"):
            results.append((k, v))
        for k, v in _linear(g("v_projection", i), f"{lp_a}.to_v"):
            results.append((k, v))
        for k, v in _linear(g("gating_query", i), f"{lp_a}.to_g"):
            results.append((k, v))
        for k, v in _linear(g("transition2", i), f"{lp_a}.out_proj"):
            results.append((k, v))
        for k, v in _linear(g("adaptive_zero_cond", i), f"{lp_a}.zero_cond"):
            results.append((k, v))

        # FFN
        ffln = g("ffw_single_cond_layer_norm", i)
        if "scale" in ffln:
            results.append((f"{lp}.ffn_cond_w.{i}", _b(ffln["scale"])))
        for k, v in _linear(g("ffw_single_cond_scale", i), f"{lp}.ffn_cond_scale.{i}"):
            results.append((k, v))
        for k, v in _linear(g("ffw_single_cond_bias", i), f"{lp}.ffn_cond_bias.{i}"):
            results.append((k, v))
        for k, v in _linear(g("ffw_transition1", i), f"{lp_f}.linear1"):
            results.append((k, v))
        for k, v in _linear(g("ffw_transition2", i), f"{lp_f}.linear2"):
            results.append((k, v))
        for k, v in _linear(g("ffw_adaptive_zero_cond", i), f"{lp_f}.zero_cond"):
            results.append((k, v))

    return results


def haiku_to_torch(
    af3_params: dict[str, dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Convert AF3 Haiku params → PyTorch state_dict.

    Returns a flat dict suitable for model.load_state_dict(..., strict=False).
    Unmapped AF3 scopes are reported via warnings.
    """
    sd: dict[str, torch.Tensor] = {}
    mapped_scopes: set[str] = set()

    def put(key: str, t: torch.Tensor):
        sd[key] = t

    def emit(gen):
        for k, t in gen:
            put(k, t)

    def get(scope: str) -> dict:
        mapped_scopes.add(scope)
        return af3_params.get(scope, {})

    def layer_slice(scope: str, idx: int) -> dict:
        """Return per-layer param dict by slicing first dim of each array."""
        raw = af3_params.get(scope, {})
        mapped_scopes.add(scope)
        return {k: v[idx] for k, v in raw.items()}

    def num_layers(scope: str) -> int:
        raw = af3_params.get(scope, {})
        if not raw:
            return 0
        return next(iter(raw.values())).shape[0]

    # Evoformer — non-stacked params
    evo = "diffuser/evoformer/"
    rel = "diffuser/evoformer/~_relative_encoding/"

    for h, pt in [
        (evo + "left_single", "evoformer.left_single"),
        (evo + "right_single", "evoformer.right_single"),
        (evo + "bond_embedding", "evoformer.bond_embedding"),
        (evo + "prev_embedding", "evoformer.prev_embedding"),
        (evo + "prev_single_embedding", "evoformer.prev_single_embedding"),
        (evo + "single_activations", "evoformer.single_activations"),
        (evo + "msa_activations", "evoformer.msa_activations"),
        (evo + "extra_msa_target_feat", "evoformer.extra_msa_target_feat"),
        (rel + "position_activations", "evoformer.position_activations"),
    ]:
        emit(_linear(get(h), pt))

    emit(_layernorm(get(evo + "prev_embedding_layer_norm"), "evoformer.prev_embedding_layer_norm"))
    emit(_layernorm(get(evo + "prev_single_embedding_layer_norm"), "evoformer.prev_single_embedding_layer_norm"))

    # MSA stack (4 layers, stacked)
    msa_base = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/"
    n_msa = num_layers(msa_base + "triangle_multiplication_outgoing/center_norm")
    for i in range(n_msa):

        def sl(sub, _i=i):
            return layer_slice(msa_base + sub, _i)

        lp = f"evoformer.msa_stack.{i}"

        emit(
            _msa_attn(
                {
                    "act_norm": sl("msa_attention1/act_norm"),
                    "pair_norm": sl("msa_attention1/pair_norm"),
                    "pair_logits": sl("msa_attention1/pair_logits"),
                    "v_projection": sl("msa_attention1/v_projection"),
                    "gating_query": sl("msa_attention1/gating_query"),
                    "output_projection": sl("msa_attention1/output_projection"),
                },
                f"{lp}.row_attn",
            )
        )

        emit(
            _outer_product(
                scope_self=sl("outer_product_mean"),
                sub={
                    "layer_norm_input": sl("outer_product_mean/layer_norm_input"),
                    "left_projection": sl("outer_product_mean/left_projection"),
                    "right_projection": sl("outer_product_mean/right_projection"),
                },
                prefix=f"{lp}.outer_product",
            )
        )

        for h_name, pt_name in [
            ("triangle_multiplication_outgoing", "tri_mul_out"),
            ("triangle_multiplication_incoming", "tri_mul_in"),
        ]:
            emit(
                _trimul(
                    {
                        "left_norm_input": sl(f"{h_name}/left_norm_input"),
                        "projection": sl(f"{h_name}/projection"),
                        "gate": sl(f"{h_name}/gate"),
                        "center_norm": sl(f"{h_name}/center_norm"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                        "gating_linear": sl(f"{h_name}/gating_linear"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        for h_name, pt_name in [("pair_attention1", "pair_attn_row"), ("pair_attention2", "pair_attn_col")]:
            emit(
                _grid_attn(
                    {
                        "act_norm": sl(f"{h_name}/act_norm"),
                        "pair_bias_projection": sl(f"{h_name}/pair_bias_projection"),
                        "q_projection": sl(f"{h_name}/q_projection"),
                        "k_projection": sl(f"{h_name}/k_projection"),
                        "v_projection": sl(f"{h_name}/v_projection"),
                        "gating_query": sl(f"{h_name}/gating_query"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        emit(
            _transition(
                {
                    "input_layer_norm": sl("pair_transition/input_layer_norm"),
                    "transition1": sl("pair_transition/transition1"),
                    "transition2": sl("pair_transition/transition2"),
                },
                f"{lp}.pair_transition",
            )
        )
        emit(
            _transition(
                {
                    "input_layer_norm": sl("msa_transition/input_layer_norm"),
                    "transition1": sl("msa_transition/transition1"),
                    "transition2": sl("msa_transition/transition2"),
                },
                f"{lp}.msa_transition",
            )
        )

    # Trunk PairFormer (48 layers, stacked)
    pf_base = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/"
    n_pf = num_layers(pf_base + "triangle_multiplication_outgoing/center_norm")
    for i in range(n_pf):

        def sl(sub, _i=i):
            return layer_slice(pf_base + sub, _i)

        lp = f"evoformer.trunk_pairformer.{i}"

        for h_name, pt_name in [
            ("triangle_multiplication_outgoing", "tri_mul_out"),
            ("triangle_multiplication_incoming", "tri_mul_in"),
        ]:
            emit(
                _trimul(
                    {
                        "left_norm_input": sl(f"{h_name}/left_norm_input"),
                        "projection": sl(f"{h_name}/projection"),
                        "gate": sl(f"{h_name}/gate"),
                        "center_norm": sl(f"{h_name}/center_norm"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                        "gating_linear": sl(f"{h_name}/gating_linear"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        for h_name, pt_name in [("pair_attention1", "pair_attn_row"), ("pair_attention2", "pair_attn_col")]:
            emit(
                _grid_attn(
                    {
                        "act_norm": sl(f"{h_name}/act_norm"),
                        "pair_bias_projection": sl(f"{h_name}/pair_bias_projection"),
                        "q_projection": sl(f"{h_name}/q_projection"),
                        "k_projection": sl(f"{h_name}/k_projection"),
                        "v_projection": sl(f"{h_name}/v_projection"),
                        "gating_query": sl(f"{h_name}/gating_query"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        emit(
            _transition(
                {
                    "input_layer_norm": sl("pair_transition/input_layer_norm"),
                    "transition1": sl("pair_transition/transition1"),
                    "transition2": sl("pair_transition/transition2"),
                },
                f"{lp}.pair_transition",
            )
        )

        # Single-track attention (single_cond=None → regular LN, no AdaLN)
        emit(
            _single_attn_flat(
                {
                    "layer_norm": sl("single_attention_layer_norm"),
                    "q_projection": sl("single_attention_q_projection"),
                    "k_projection": sl("single_attention_k_projection"),
                    "v_projection": sl("single_attention_v_projection"),
                    "gating_query": sl("single_attention_gating_query"),
                    "transition2": sl("single_attention_transition2"),
                },
                f"{lp}.single_attn",
            )
        )

        emit(_layernorm(sl("single_pair_logits_norm"), f"{lp}.single_to_pair_norm"))
        emit(_linear(sl("single_pair_logits_projection"), f"{lp}.single_to_pair"))
        emit(
            _transition(
                {
                    "input_layer_norm": sl("single_transition/input_layer_norm"),
                    "transition1": sl("single_transition/transition1"),
                    "transition2": sl("single_transition/transition2"),
                },
                f"{lp}.single_transition",
            )
        )

    # Confidence head (4 layers stacked)
    conf_pf = "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer/"
    n_conf = num_layers(conf_pf + "triangle_multiplication_outgoing/center_norm")
    for i in range(n_conf):

        def sl(sub, _i=i):
            return layer_slice(conf_pf + sub, _i)

        lp = f"confidence_head.layers.{i}"

        for h_name, pt_name in [
            ("triangle_multiplication_outgoing", "tri_mul_out"),
            ("triangle_multiplication_incoming", "tri_mul_in"),
        ]:
            emit(
                _trimul(
                    {
                        "left_norm_input": sl(f"{h_name}/left_norm_input"),
                        "projection": sl(f"{h_name}/projection"),
                        "gate": sl(f"{h_name}/gate"),
                        "center_norm": sl(f"{h_name}/center_norm"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                        "gating_linear": sl(f"{h_name}/gating_linear"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        for h_name, pt_name in [("pair_attention1", "pair_attn_row"), ("pair_attention2", "pair_attn_col")]:
            emit(
                _grid_attn(
                    {
                        "act_norm": sl(f"{h_name}/act_norm"),
                        "pair_bias_projection": sl(f"{h_name}/pair_bias_projection"),
                        "q_projection": sl(f"{h_name}/q_projection"),
                        "k_projection": sl(f"{h_name}/k_projection"),
                        "v_projection": sl(f"{h_name}/v_projection"),
                        "gating_query": sl(f"{h_name}/gating_query"),
                        "output_projection": sl(f"{h_name}/output_projection"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        emit(
            _transition(
                {
                    "input_layer_norm": sl("pair_transition/input_layer_norm"),
                    "transition1": sl("pair_transition/transition1"),
                    "transition2": sl("pair_transition/transition2"),
                },
                f"{lp}.pair_transition",
            )
        )

        # Confidence head also has single track
        emit(
            _single_attn_flat(
                {
                    "layer_norm": sl("single_attention_layer_norm"),
                    "q_projection": sl("single_attention_q_projection"),
                    "k_projection": sl("single_attention_k_projection"),
                    "v_projection": sl("single_attention_v_projection"),
                    "gating_query": sl("single_attention_gating_query"),
                    "transition2": sl("single_attention_transition2"),
                },
                f"{lp}.single_attn",
            )
        )

        emit(_layernorm(sl("single_pair_logits_norm"), f"{lp}.single_to_pair_norm"))
        emit(_linear(sl("single_pair_logits_projection"), f"{lp}.single_to_pair"))
        emit(
            _transition(
                {
                    "input_layer_norm": sl("single_transition/input_layer_norm"),
                    "transition1": sl("single_transition/transition1"),
                    "transition2": sl("single_transition/transition2"),
                },
                f"{lp}.single_transition",
            )
        )

    # Confidence head _embed_features
    conf_emb = "diffuser/confidence_head/~_embed_features/"
    emit(_linear(get(conf_emb + "left_target_feat_project"), "confidence_head.left_target_feat_project"))
    emit(_linear(get(conf_emb + "right_target_feat_project"), "confidence_head.right_target_feat_project"))
    emit(_linear(get(conf_emb + "distogram_feat_project"), "confidence_head.distogram_feat_project"))

    # Confidence head output projections (with LayerNorms)
    conf = "diffuser/confidence_head/"
    emit(_layernorm(get(conf + "plddt_logits_ln"), "confidence_head.plddt_logits_ln"))
    emit(_linear(get(conf + "plddt_logits"), "confidence_head.plddt_logits"))
    emit(_layernorm(get(conf + "pae_logits_ln"), "confidence_head.pae_logits_ln"))
    emit(_linear(get(conf + "pae_logits"), "confidence_head.pae_logits"))

    # PDE head (logits_ln in AF3 = pde_logits_ln, left_half_distance_logits = pde_half_logits)
    emit(_layernorm(get(conf + "logits_ln"), "confidence_head.pde_logits_ln"))
    emit(_linear(get(conf + "left_half_distance_logits"), "confidence_head.pde_half_logits"))

    # Experimentally resolved head
    emit(_layernorm(get(conf + "experimentally_resolved_ln"), "confidence_head.exp_resolved_ln"))
    emit(_linear(get(conf + "experimentally_resolved_logits"), "confidence_head.exp_resolved_logits"))

    # Distogram head
    disto = "diffuser/distogram_head/"
    emit(_linear(get(disto + "half_logits"), "distogram_head.half_logits"))

    # Diffusion head — conditioning
    dh = "diffuser/~/diffusion_head/"

    # Scale-only norms (create_offset=False in AF3) → bare nn.Parameter
    emit(_scale_param(get(dh + "pair_cond_initial_norm"), "diffusion_head.conditioning.pair_cond_norm_w"))
    emit(_linear(get(dh + "pair_cond_initial_projection"), "diffusion_head.conditioning.pair_cond_proj"))
    emit(_scale_param(get(dh + "single_cond_initial_norm"), "diffusion_head.conditioning.single_cond_norm_w"))
    emit(_linear(get(dh + "single_cond_initial_projection"), "diffusion_head.conditioning.single_cond_proj"))
    emit(_scale_param(get(dh + "noise_embedding_initial_norm"), "diffusion_head.conditioning.noise_emb_norm_w"))
    emit(_linear(get(dh + "noise_embedding_initial_projection"), "diffusion_head.conditioning.noise_emb_proj"))

    for idx in range(2):
        pp = dh + f"pair_transition_{idx}"
        sp = dh + f"single_transition_{idx}"
        emit(_layernorm(get(pp + "ffw_layer_norm"), f"diffusion_head.conditioning.pair_trans{idx}.norm"))
        emit(_linear(get(pp + "ffw_transition1"), f"diffusion_head.conditioning.pair_trans{idx}.linear1"))
        emit(_linear(get(pp + "ffw_transition2"), f"diffusion_head.conditioning.pair_trans{idx}.linear2"))
        emit(_layernorm(get(sp + "ffw_layer_norm"), f"diffusion_head.conditioning.single_trans{idx}.norm"))
        emit(_linear(get(sp + "ffw_transition1"), f"diffusion_head.conditioning.single_trans{idx}.linear1"))
        emit(_linear(get(sp + "ffw_transition2"), f"diffusion_head.conditioning.single_trans{idx}.linear2"))

    # Single cond embedding (between encoder and transformer) — scale-only norm
    emit(_scale_param(get(dh + "single_cond_embedding_norm"), "diffusion_head.single_cond_emb_norm_w"))
    emit(_linear(get(dh + "single_cond_embedding_projection"), "diffusion_head.single_cond_embedding_projection"))

    # Diffusion main transformer (24 blocks = 6 super × 4)
    trans_outer = dh + "transformer/__layer_stack_with_per_layer/"
    trans_inner = trans_outer + "__layer_stack_with_per_layer/transformer"

    emit(_scale_param(get(dh + "transformer/pair_input_layer_norm"), "diffusion_head.transformer.pair_norm_w"))
    emit(_scale_param(get(dh + "output_norm"), "diffusion_head.transformer.output_norm_w"))
    # pair_logits_proj: stacked [n_super, in, sbs, H] → per-super-block Linear
    plp_raw = af3_params.get(trans_outer + "pair_logits_projection", {})
    mapped_scopes.add(trans_outer + "pair_logits_projection")
    if "weights" in plp_raw:
        for si in range(plp_raw["weights"].shape[0]):
            put(f"diffusion_head.transformer.pair_logits_proj.{si}.weight", _w(plp_raw["weights"][si]))

    # Per-block params (shape [n_super, n_block_per_super, ...])
    q_raw = af3_params.get(trans_inner + "q_projection", {})
    if q_raw and "weights" in q_raw:
        n_super, n_bps = q_raw["weights"].shape[:2]

        def get_trans(param, s, b):
            raw = af3_params.get(trans_inner + param, {})
            mapped_scopes.add(trans_inner + param)
            return {k: v[s, b] for k, v in raw.items()}

        for s in range(n_super):
            for b in range(n_bps):
                li = s * n_bps + b
                lp = f"diffusion_head.transformer.layers.{li}"

                # AdaLN attention
                sc_ln = get_trans("single_cond_layer_norm", s, b)
                if "scale" in sc_ln:
                    put(f"{lp}.attn_cond_norm_w", _b(sc_ln["scale"]))
                emit(_linear(get_trans("single_cond_scale", s, b), f"{lp}.attn_cond_scale"))
                emit(_linear(get_trans("single_cond_bias", s, b), f"{lp}.attn_cond_bias"))
                emit(_linear(get_trans("q_projection", s, b), f"{lp}.to_q"))
                emit(_linear(get_trans("k_projection", s, b), f"{lp}.to_k"))
                emit(_linear(get_trans("v_projection", s, b), f"{lp}.to_v"))
                emit(_linear(get_trans("gating_query", s, b), f"{lp}.to_g"))
                emit(_linear(get_trans("transition2", s, b), f"{lp}.out_proj"))
                emit(_linear(get_trans("adaptive_zero_cond", s, b), f"{lp}.zero_cond"))

                # AdaLN FFN
                ffw_ln = get_trans("ffw_single_cond_layer_norm", s, b)
                if "scale" in ffw_ln:
                    put(f"{lp}.ffn_cond_norm_w", _b(ffw_ln["scale"]))
                emit(_linear(get_trans("ffw_single_cond_scale", s, b), f"{lp}.ffn_cond_scale"))
                emit(_linear(get_trans("ffw_single_cond_bias", s, b), f"{lp}.ffn_cond_bias"))
                emit(_linear(get_trans("ffw_transition1", s, b), f"{lp}.ffn_linear1"))
                emit(_linear(get_trans("ffw_transition2", s, b), f"{lp}.ffn_linear2"))
                emit(_linear(get_trans("ffw_adaptive_zero_cond", s, b), f"{lp}.ffn_zero_cond"))

    # Diffusion head — per-atom conditioning (shared between enc/dec)
    enc_p = dh + "diffusion_"

    def _per_atom_cond(src: str, pt: str):
        emit(_linear(get(src + "embed_ref_pos"), f"{pt}.per_atom.embed_ref_pos"))
        emit(_linear(get(src + "embed_ref_mask"), f"{pt}.per_atom.embed_ref_mask"))
        emit(_linear(get(src + "embed_ref_element"), f"{pt}.per_atom.embed_ref_element"))
        emit(_linear(get(src + "embed_ref_charge"), f"{pt}.per_atom.embed_ref_charge"))
        emit(_linear(get(src + "embed_ref_atom_name"), f"{pt}.per_atom.embed_ref_atom_name"))
        emit(_linear(get(src + "single_to_pair_cond_row"), f"{pt}.per_atom.single_to_pair_row"))
        emit(_linear(get(src + "single_to_pair_cond_col"), f"{pt}.per_atom.single_to_pair_col"))
        emit(_linear(get(src + "embed_pair_offsets"), f"{pt}.per_atom.embed_pair_offsets"))
        emit(_linear(get(src + "embed_pair_distances"), f"{pt}.per_atom.embed_pair_distances"))
        emit(_linear(get(src + "pair_mlp_1"), f"{pt}.per_atom.pair_mlp1"))
        emit(_linear(get(src + "pair_mlp_2"), f"{pt}.per_atom.pair_mlp2"))
        emit(_linear(get(src + "pair_mlp_3"), f"{pt}.per_atom.pair_mlp3"))

    _per_atom_cond(enc_p, "diffusion_head.atom_encoder")

    # Encoder trunk conditioning (scale-only LayerNorm → bare Parameter)
    emit(_scale_param(get(enc_p + "lnorm_trunk_single_cond"), "diffusion_head.atom_encoder.lnorm_trunk_single_w"))
    emit(_linear(get(enc_p + "embed_trunk_single_cond"), "diffusion_head.atom_encoder.embed_trunk_single"))
    emit(_scale_param(get(enc_p + "lnorm_trunk_pair_cond"), "diffusion_head.atom_encoder.lnorm_trunk_pair_w"))
    emit(_linear(get(enc_p + "embed_trunk_pair_cond"), "diffusion_head.atom_encoder.embed_trunk_pair"))

    # Encoder secondary pair cond (index _1 variants)
    emit(_linear(get(enc_p + "single_to_pair_cond_row_1"), "diffusion_head.atom_encoder.single_to_pair_row_enc"))
    emit(_linear(get(enc_p + "single_to_pair_cond_col_1"), "diffusion_head.atom_encoder.single_to_pair_col_enc"))
    emit(_linear(get(enc_p + "embed_pair_offsets_1"), "diffusion_head.atom_encoder.enc_embed_pair_offsets"))
    emit(_linear(get(enc_p + "embed_pair_distances_1"), "diffusion_head.atom_encoder.enc_embed_pair_distances"))
    emit(_linear(get(enc_p + "embed_pair_offsets_valid"), "diffusion_head.atom_encoder.enc_embed_offsets_valid"))
    emit(_linear(get(enc_p + "atom_positions_to_features"), "diffusion_head.atom_encoder._pos_proj"))
    emit(_linear(get(enc_p + "project_atom_features_for_aggr"), "diffusion_head.atom_encoder.proj_for_aggr"))

    # Atom encoder transformer
    for k, v in _atom_cross_att_transformer(
        af3_params,
        dh + "diffusion_atom_transformer_encoder/",
        "diffusion_head.atom_encoder.atom_transformer",
        mark_fn=mapped_scopes.add,
    ):
        put(k, v)

    # Decoder
    emit(_linear(get(enc_p + "project_token_features_for_broadcast"), "diffusion_head.atom_decoder.proj_tokens"))
    emit(_scale_param(get(enc_p + "atom_features_layer_norm"), "diffusion_head.atom_decoder.atom_norm_w"))
    emit(_linear(get(enc_p + "atom_features_to_position_update"), "diffusion_head.atom_decoder.pos_update"))

    for k, v in _atom_cross_att_transformer(
        af3_params,
        dh + "diffusion_atom_transformer_decoder/",
        "diffusion_head.atom_decoder.atom_transformer",
        mark_fn=mapped_scopes.add,
    ):
        put(k, v)

    # Evoformer conditioning encoder (produces target_feat contribution)
    # Scope prefix: "diffuser/evoformer_conditioning_"
    evo_p = "diffuser/evoformer_conditioning_"

    _per_atom_cond(evo_p, "evo_cond_encoder")

    # Encoder-specific pair conditioning (_1 variants)
    emit(_linear(get(evo_p + "single_to_pair_cond_row_1"), "evo_cond_encoder.single_to_pair_row_enc"))
    emit(_linear(get(evo_p + "single_to_pair_cond_col_1"), "evo_cond_encoder.single_to_pair_col_enc"))
    emit(_linear(get(evo_p + "embed_pair_offsets_1"), "evo_cond_encoder.enc_embed_pair_offsets"))
    emit(_linear(get(evo_p + "embed_pair_distances_1"), "evo_cond_encoder.enc_embed_pair_distances"))
    emit(_linear(get(evo_p + "embed_pair_offsets_valid"), "evo_cond_encoder.enc_embed_offsets_valid"))
    emit(_linear(get(evo_p + "project_atom_features_for_aggr"), "evo_cond_encoder.proj_for_aggr"))

    # Atom transformer for evoformer conditioning
    for k, v in _atom_cross_att_transformer(
        af3_params,
        "diffuser/evoformer_conditioning_atom_transformer_encoder/",
        "evo_cond_encoder.atom_transformer",
        mark_fn=mapped_scopes.add,
    ):
        put(k, v)

    # Template embedding
    # Haiku scope: diffuser/evoformer/template_embedding/
    #   output_linear                         → evoformer.template_embedding.output_proj
    #   single_template_embedding/
    #     query_embedding_norm                → ...single_template_emb.query_norm
    #     template_pair_embedding_0..8        → ...single_template_emb.embed_{dgram,...,query}
    #     output_layer_norm                   → ...single_template_emb.output_norm
    #     __layer_stack_no_per_layer/template_embedding_iteration/
    #       (2-layer PairFormer, no single track)
    te = "diffuser/evoformer/template_embedding/"
    ste = te + "single_template_embedding/"
    te_pt = "evoformer.template_embedding"
    ste_pt = f"{te_pt}.single_template_emb"

    # Output projection (TemplateEmbedding.output_proj)
    emit(_linear(get(te + "output_linear"), f"{te_pt}.output_proj"))

    # SingleTemplateEmbedding: query_norm, 9 embedding projections, output_norm
    emit(_layernorm(get(ste + "query_embedding_norm"), f"{ste_pt}.query_norm"))

    # 9 template pair embedding projections.
    # Indices 1, 4, 5, 6, 7 use num_input_dims=0 in AF3 (scalar inputs:
    # pseudo_beta_mask, unit_vec x/y/z, backbone_mask). Their Haiku weights
    # have shape (num_channels,) instead of (in_features, num_channels).
    embed_names = [
        ("embed_dgram", False),  # 0: [N,N,39] → 64, num_input_dims=1
        ("embed_pseudo_beta", True),  # 1: [N,N] scalar → 64, num_input_dims=0
        ("embed_aatype_row", False),  # 2: [1,N,31] → 64, num_input_dims=1
        ("embed_aatype_col", False),  # 3: [N,1,31] → 64, num_input_dims=1
        ("embed_unit_vec_x", True),  # 4: [N,N] scalar → 64, num_input_dims=0
        ("embed_unit_vec_y", True),  # 5: [N,N] scalar → 64, num_input_dims=0
        ("embed_unit_vec_z", True),  # 6: [N,N] scalar → 64, num_input_dims=0
        ("embed_backbone_mask", True),  # 7: [N,N] scalar → 64, num_input_dims=0
        ("embed_query", False),  # 8: [N,N,128] → 64, num_input_dims=1
    ]
    for i, (pt_name, is_scalar) in enumerate(embed_names):
        if is_scalar:
            emit(_linear_scalar(get(ste + f"template_pair_embedding_{i}"), f"{ste_pt}.{pt_name}"))
        else:
            emit(_linear(get(ste + f"template_pair_embedding_{i}"), f"{ste_pt}.{pt_name}"))

    emit(_layernorm(get(ste + "output_layer_norm"), f"{ste_pt}.output_norm"))

    # Template PairFormer stack (2 layers, stacked via layer_stack_no_per_layer)
    te_pf_base = ste + "__layer_stack_no_per_layer/template_embedding_iteration/"
    # Detect number of layers from any stacked param
    te_pf_test_scope = te_pf_base + "triangle_multiplication_outgoing/center_norm"
    n_te_pf = num_layers(te_pf_test_scope)
    for i in range(n_te_pf):

        def sl_te(sub, _i=i):
            return layer_slice(te_pf_base + sub, _i)

        lp = f"{ste_pt}.pairformer.{i}"

        for h_name, pt_name in [
            ("triangle_multiplication_outgoing", "tri_mul_out"),
            ("triangle_multiplication_incoming", "tri_mul_in"),
        ]:
            emit(
                _trimul(
                    {
                        "left_norm_input": sl_te(f"{h_name}/left_norm_input"),
                        "projection": sl_te(f"{h_name}/projection"),
                        "gate": sl_te(f"{h_name}/gate"),
                        "center_norm": sl_te(f"{h_name}/center_norm"),
                        "output_projection": sl_te(f"{h_name}/output_projection"),
                        "gating_linear": sl_te(f"{h_name}/gating_linear"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        for h_name, pt_name in [("pair_attention1", "pair_attn_row"), ("pair_attention2", "pair_attn_col")]:
            emit(
                _grid_attn(
                    {
                        "act_norm": sl_te(f"{h_name}/act_norm"),
                        "pair_bias_projection": sl_te(f"{h_name}/pair_bias_projection"),
                        "q_projection": sl_te(f"{h_name}/q_projection"),
                        "k_projection": sl_te(f"{h_name}/k_projection"),
                        "v_projection": sl_te(f"{h_name}/v_projection"),
                        "gating_query": sl_te(f"{h_name}/gating_query"),
                        "output_projection": sl_te(f"{h_name}/output_projection"),
                    },
                    f"{lp}.{pt_name}",
                )
            )

        emit(
            _transition(
                {
                    "input_layer_norm": sl_te("pair_transition/input_layer_norm"),
                    "transition1": sl_te("pair_transition/transition1"),
                    "transition2": sl_te("pair_transition/transition2"),
                },
                f"{lp}.pair_transition",
            )
        )

    # Report unmapped scopes
    skip_prefixes = ("__meta__",)
    for scope in sorted(af3_params.keys()):
        if scope in mapped_scopes:
            continue
        if any(scope.startswith(p) or scope == p for p in skip_prefixes):
            continue
        warnings.warn(f"[af3 converter] unmapped scope: {scope}", stacklevel=2)

    return sd


def extract_fourier_weights(af3_ref_dir: Path) -> dict[str, torch.Tensor]:
    """Extract Fourier embedding constants from the AF3 reference source.

    Reads ``noise_level_embeddings.py`` and parses the ``_WEIGHT`` and
    ``_BIAS`` list literals using :func:`ast.literal_eval`.

    Returns:
        Dict with keys ``diffusion_head.conditioning.fourier_weight`` and
        ``diffusion_head.conditioning.fourier_bias``, each a float32 tensor
        of shape [256].
    """
    src = af3_ref_dir / "src" / "alphafold3" / "model" / "network" / "noise_level_embeddings.py"
    text = src.read_text()

    def _parse_list(var_name: str) -> list[float]:
        pattern = rf"^{var_name}\s*=\s*\["
        m = re.search(pattern, text, re.MULTILINE)
        if m is None:
            raise ValueError(f"Could not find {var_name} in {src}")
        start = m.start() + text[m.start() :].index("[")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    return ast.literal_eval(text[start : i + 1])
        raise ValueError(f"Unmatched bracket for {var_name} in {src}")

    weight = _parse_list("_WEIGHT")
    bias = _parse_list("_BIAS")

    return {
        "diffusion_head.conditioning.fourier_weight": torch.tensor(weight, dtype=torch.float32),
        "diffusion_head.conditioning.fourier_bias": torch.tensor(bias, dtype=torch.float32),
    }
