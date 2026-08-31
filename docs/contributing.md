# Contributing & Extending

Guide for developers working on DeepFold3: setup, conventions, testing, and extension patterns.

## Development Setup

```bash
git clone <repo-url> && cd deepfold3-dev
uv sync --dev
uv run pytest                    # run all tests
uv run pytest -m "not slow"      # skip expensive tests
uv run ruff check src/           # lint
uv run ruff format src/          # format
```

Requirements: Python 3.11+ (pinned in `.python-version`), `uv` package manager.

## Code Style

**Ruff**: line length 120, double quotes, space indent. Rules: E, F, I (isort), B (bugbear), UP (pyupgrade). `E501` is ignored (handled by formatter).

**Conventions**:

- `LinearNoBias` consistently used instead of `nn.Linear(bias=False)` — matches AF3's `hm.Linear` default
- Type hints on all function signatures
- Docstrings: Google style (`Args:`, `Returns:`, `Raises:` sections). Module-level (purpose + Haiku scope mapping), class-level (`Attributes:` for dataclasses), function-level (args/returns for public API)

## Testing

### Running Tests

```bash
uv run pytest                                # all tests, CPU
uv run pytest --device cuda                  # GPU
uv run pytest --device cuda:1                # specific GPU
uv run pytest tests/test_evoformer.py        # single file
uv run pytest tests/test_ops.py::TestTriangleMul  # single class
uv run pytest -m "not slow"                  # skip slow tests
```

### Test Structure

14 test files, ~190 tests total:

| File                     | Purpose                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| `test_primitives.py`     | LinearNoBias, LayerNorm, SwiGLU, TransitionBlock, AdaLayerNorm   |
| `test_attention.py`      | GridSelfAttention (row/col), MSAAttention, SingleAttention       |
| `test_ops.py`            | TriangleMul (incoming/outgoing), OuterProductMean                |
| `test_embedders.py`      | Relative encoding, profile, deletion mean                        |
| `test_evoformer.py`      | EvoformerBlock, PairFormerBlock, full Evoformer                  |
| `test_diffusion.py`      | Noise schedule, Fourier embedding, random augmentation           |
| `test_jax_comparison.py` | PyTorch vs JAX/Haiku numerical validation                        |
| `test_weight_loading.py` | AF3 weight converter: missing/unexpected keys, shapes            |
| `test_e2e.py`            | Full forward pass, distogram properties, data pipeline           |
| `test_ligand.py`         | CCD molecule loading, SMILES, JSON parsing, ligand featurization |

### Fixtures (`conftest.py`)

Session-scoped (loaded once):

- `device` — from `--device` CLI option (default: `"cpu"`)
- `af3_raw_params` — raw AF3 Haiku params from `resources/af3.bin.zst` (skips if missing)
- `af3_state_dict` — converted PyTorch state_dict
- `model` — full DeepFold3 in eval mode on device

Function-scoped (fresh per test):

- `small_features` — 8-residue protein features (N=8, M=24, S=4)
- `medium_features` — 32-residue protein features (N=32, M=24, S=16)
- `rng_generator` — deterministic `torch.Generator` (seed 42)
- `np_rng` — deterministic numpy RNG (seed 42)

### Test Patterns

**Naming**: `test_<module>_<behavior>` or class-based `Test<Module>::test_<behavior>`

**Reference implementations**: most modules include manual reference implementations (using `F.linear`, `torch.matmul`) to validate forward pass correctness.

**JAX comparison**: critical for architecture equivalence. Helper functions convert between PyTorch and Haiku weight layouts (`_haiku_w` transposes `[out, in]` → `[in, out]`).

Coverage is per-module, and the gap matters as much as the tests: the trunk's
building blocks (LayerNorm, TransitionBlock, TriangleMul, OuterProductMean, the
three attentions, one PairFormer layer) and the diffusion transformer
(`AdaLayerNorm`, `AdaTransitionBlock`, diffusion self-attention with mask + pair
bias, and the assembled `DiffusionTransformerBlock`) are pinned against a JAX
reimplementation of the AF3 reference. `atom_cross_attn`, `template_embedding`,
`confidence`, and the assembled forward with recycling are NOT — a converted-AF3
divergence can still live there.

Randomise before comparing (`_randomise`): AdaLN-zero ships several projections
zero-initialised and its gate biases at -2, so a freshly built block is partly a
no-op and a wrong wiring passes by multiplying zeros. New parity tests should be
mutation-checked — break the wiring on purpose and confirm the test fails —
since a test that cannot fail adds coverage only on paper.

**Batch support**: explicit tests verifying unbatched `[N, ...]` and batched `[B, N, ...]` produce consistent results.

**Markers**:

- `@pytest.mark.slow` — expensive tests (full forward pass, multi-sample inference)
- `@pytest.mark.network` — tests requiring network access (CCD downloads from RCSB)
- `@pytest.mark.timeout(120)` — override global 60s timeout for slow tests

## How to Add a New Model Head

Using `DistogramHead` as the simplest reference:

### 1. Add config (`model/config.py`)

```python
@dataclasses.dataclass
class MyHeadConfig:
    num_bins: int = 64
```

Add it as a field in `ModelConfig`.

### 2. Implement the module

```python
# model/my_head.py (or add to model.py for simple heads)
class MyHead(nn.Module):
    """My head description.

    Haiku scope: ``my_head``
        ``logits``: LinearNoBias(pair_channels, num_bins)
    """
    def __init__(self, pair_channels: int, num_bins: int = 64):
        super().__init__()
        self.logits = LinearNoBias(pair_channels, num_bins)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.logits(pair)
```

### 3. Register in `DeepFold3` (`model/model.py`)

Add to `__init__()`:

```python
self.my_head = MyHead(cfg.evoformer.pair_channel, cfg.my_head.num_bins)
```

Wire in `forward()` and add output field to `DeepFold3Output`.

### 4. Add weight mapping (`convert/af3.py`)

Add the Haiku scope → PyTorch key mapping to the conversion table. Pattern: Haiku stores weights as `scope/name` with shape `[in, out]`; PyTorch uses `module.param` with shape `[out, in]`.

### 5. Add tests

- Shape and dtype validation with random weights
- Numerical comparison with AF3 weights (if available)
- Batch dimension support

## Weight Conversion Guide

`convert/af3.py` maps AF3 Haiku parameter scopes to PyTorch state_dict keys.

**Key patterns**:

- Haiku `hm.Linear` weight `[in, out]` → PyTorch `LinearNoBias` weight `[out, in]` (transposed)
- Haiku uses `/` scope separators; PyTorch uses `.` module separators
- Each module's docstring documents its Haiku attribute names

**`skip_prefixes`**: lists Haiku scopes for unimplemented modules. These are silently skipped during conversion.

**Validation**: `test_weight_loading.py` loads AF3 params and checks for:

- Missing keys (in PyTorch model but not in converted params)
- Unexpected keys (in converted params but not in PyTorch model)
- Shape mismatches between converted and expected

## Debugging Tips

**Shape mismatches**: Check `has_batch` dimension handling. All core modules detect unbatched vs batched via `x.dim()` and unsqueeze/squeeze accordingly.

**Diffusion output quality**: Ensure `_random_augmentation()` is called at every denoising step. The model was trained with center/rotate/translate per step; without it, output is garbage.

**Confidence metrics**: `atom_mask` is `[N, M]` but confidence outputs are `[num_samples, N, ...]`. Use `atom_mask` (not `atom_mask[0]`) for masking.

**pLDDT reshape**: Raw output is `[N, M*50]` — reshape to `[N, M, 50]` before softmax, not as a flat 1200-class distribution.

**Quick feature checks**:

```bash
uv run python -c "
from deepfold3.data.module.dataset import InferenceInput
feats = InferenceInput.from_sequence('MKTIIALSYIFCLVFA')
print({k: tuple(v.shape) for k, v in feats.items()})
"
```

**GridSelfAttention pair bias**: The bias is non-batched `[1, H, N, N]` — the same bias applies to all rows. If you see per-row bias indexing, that's a bug.

## Performance Considerations

**Compute bottlenecks** (in order):

1. **Diffusion**: 200 steps x 24 transformer blocks x atom cross-att encode/decode
2. **Evoformer**: 48 PairFormer layers dominate trunk compute
3. **Atom cross-attention**: Q=32/K=128 subset sizes control memory/compute tradeoff

**Memory tips**:

- `torch.no_grad()` for inference
- `.detach()` between recycles prevents gradient accumulation
- `atom_cross_attn` is not yet batch-aware — subset building is per-sample

**Subset sizes**: `queries_subset_size=32` and `keys_subset_size=128` in `AtomCrossAttConfig` control the atom cross-attention window. Larger K gives better spatial coverage but increases memory quadratically within each subset.


## Zero-init output paths make comparison tests vacuous

Every AF3-style module zero-initializes its output projection (`out_proj`,
`out_gate`, `linear2`, AdaLN-zero conditioners), so a freshly constructed one
emits exactly zero. A test of the form

```python
assert_close(module(x), hand_written_reference(x))
```

then compares zero with zero and passes whatever the module did in between — the
reference body (interleaved channel split, einsum direction, LayerNorm
placement) is never exercised. A missing `head_norm == "channel"` branch in
MHR-TriMul survived a test named `..._is_exactly_af3` exactly this way, and a
sweep found 46 such comparison sites across the suite.

Call `wake_zero_init` from `tests/_util.py` on any module whose output you are
about to compare, and assert the compared tensor is non-zero:

```python
from tests._util import wake_zero_init

mod = wake_zero_init(TriangleMulOutgoing(c_z=32))
out = mod(z, mask)
assert out.abs().max() > 1e-3
```

Run `uv run pytest --check-vacuous` to list comparisons where both sides are
all zero; the report is written to `tmp/vacuous_hits.txt`. Four remain and are
legitimate — they verify that a result *is* zero.

Related: `torch.autograd.gradcheck` in fp64 does not work on any module using
`primitives.LayerNorm`, which always runs its forward in fp32; under `.double()`
the finite differences are swamped by fp32 rounding. Compare analytic gradients
against a reference implementation instead.
