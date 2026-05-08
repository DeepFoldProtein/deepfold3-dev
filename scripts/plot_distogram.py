"""Plot a pairwise representative-atom distance map (distogram) from a structure.

Source can be a local mmCIF (predicted or ground-truth) or an RCSB PDB ID
fetched on the fly. Representative atom per token: Cα for protein, C1' for
nucleic acid, every heavy atom for ligands (one token per atom).

Usage
-----
    python scripts/plot_distogram.py --cif path/to/model.cif --output out.png
    python scripts/plot_distogram.py --pdb-id 7QP5 --output out.png
    python scripts/plot_distogram.py --cif a.cif --cif b.cif --output cmp.png
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import re
import urllib.request
from pathlib import Path

import gemmi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepfold3.model.config import DistogramConfig

REP_PROTEIN = "CA"
REP_NUCLEIC = "C1'"

_NATSORT_RE = re.compile(r"(\d+)")


def _natkey(s: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in _NATSORT_RE.split(s))


def _load_predicted_distogram(
    path: Path, chain_filter: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load expected pairwise distance map from a `confidences.json`.

    Returns (E[d] [N, N] in Å, chains [N]). When ``chain_filter`` is given,
    keeps only tokens whose ``token_chain_ids`` matches.
    """
    with open(path) as f:
        d = json.load(f)
    if "distogram" not in d or "token_chain_ids" not in d:
        raise ValueError(
            f"{path} missing keys (need 'distogram' + 'token_chain_ids'); "
            "regenerate with current deepfold3 inference"
        )
    chains_all = np.asarray(d["token_chain_ids"], dtype=str)
    expected = np.asarray(d["distogram"], dtype=np.float64)
    if expected.ndim != 2 or expected.shape[0] != expected.shape[1] or expected.shape[0] != len(chains_all):
        raise ValueError(f"{path}: distogram shape {expected.shape} inconsistent with chain count {len(chains_all)}")
    if chain_filter is not None:
        mask = chains_all == chain_filter
        if not mask.any():
            raise ValueError(f"{path}: no tokens with chain {chain_filter!r}; available: {sorted(set(chains_all))}")
        expected = expected[np.ix_(mask, mask)]
        chains_all = chains_all[mask]
    return expected, chains_all.tolist()


def _read_structure(path: Path) -> gemmi.Structure:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            doc = gemmi.cif.read_string(fh.read())
        st = gemmi.make_structure_from_block(doc.sole_block())
    else:
        st = gemmi.read_structure(str(path))
    st.setup_entities()
    return st


def _fetch_pdb(pdb_id: str, cache_dir: Path) -> Path:
    pdb_id = pdb_id.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id}.cif.gz"
    if not out.exists():
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        print(f"[plot_distogram] fetching {url}")
        urllib.request.urlretrieve(url, out)
    return out


_KIND_OPTIONS = ("protein", "dna", "rna", "ligand")


def extract_tokens(
    st: gemmi.Structure,
    chain_filter: str | None = None,
    kind_filter: set[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], str]:
    """Return (xyz [N,3], chain_ids [N], kinds [N], seq) for representative-atom tokens.

    ``seq`` is a per-token 1-letter string (X for ligand atoms or unknown polymer residues).

    If ``chain_filter`` is set, only include tokens whose chain name matches.
    If ``kind_filter`` is set, only include tokens whose kind ∈ {protein, dna, rna, ligand}.
    """
    coords: list[tuple[float, float, float]] = []
    chains: list[str] = []
    kinds: list[str] = []
    seq_chars: list[str] = []
    for chain in st[0]:
        if chain_filter is not None and chain.name != chain_filter:
            continue
        polymer = chain.get_polymer()
        if polymer:
            ptype = polymer.check_polymer_type()
            if ptype == gemmi.PolymerType.Dna:
                kind, rep = "dna", REP_NUCLEIC
            elif ptype == gemmi.PolymerType.Rna:
                kind, rep = "rna", REP_NUCLEIC
            elif ptype == gemmi.PolymerType.DnaRnaHybrid:
                kind, rep = "rna", REP_NUCLEIC
            else:
                kind, rep = "protein", REP_PROTEIN
            if kind_filter is None or kind in kind_filter:
                for res in polymer:
                    atom = res.find_atom(rep, "*")
                    if atom is None:
                        continue
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    chains.append(chain.name)
                    kinds.append(kind)
                    seq_chars.append(gemmi.find_tabulated_residue(res.name).one_letter_code.upper() or "X")
        if kind_filter is None or "ligand" in kind_filter:
            for res in chain:
                if res.entity_type == gemmi.EntityType.NonPolymer:
                    for atom in res:
                        if atom.element.name == "H":
                            continue
                        coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                        chains.append(chain.name)
                        kinds.append("ligand")
                        seq_chars.append("X")
    if not coords:
        raise ValueError("no representative atoms found in structure")
    return np.asarray(coords, dtype=np.float64), chains, kinds, "".join(seq_chars)


def _align_substr(src_seq: str, ref_seq: str) -> tuple[int, int, int] | None:
    """If one sequence is a contiguous substring of the other, return slice indices.

    Returns (src_offset, ref_offset, length) such that
    ``src_seq[src_offset:src_offset+length] == ref_seq[ref_offset:ref_offset+length]``.
    Returns None when no substring containment match exists.
    """
    if not src_seq or not ref_seq:
        return None
    if ref_seq in src_seq:
        return src_seq.find(ref_seq), 0, len(ref_seq)
    if src_seq in ref_seq:
        return 0, ref_seq.find(src_seq), len(src_seq)
    return None


def chain_boundaries(chains: list[str]) -> list[int]:
    bounds: list[int] = []
    for i in range(1, len(chains)):
        if chains[i] != chains[i - 1]:
            bounds.append(i)
    return bounds


def plot_panel(
    ax: plt.Axes,
    D: np.ndarray,
    chains: list[str],
    title: str,
    vmax: float,
    cmap: str,
) -> None:
    cmap_obj = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#bbbbbb")  # masked / dropped residues
    M = np.ma.masked_invalid(D)
    im = ax.imshow(M, origin="upper", cmap=cmap_obj, vmin=0.0, vmax=vmax, aspect="equal")
    for b in chain_boundaries(chains):
        ax.axhline(b - 0.5, color="white", lw=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    plt.colorbar(im, ax=ax, shrink=0.82, label="Å")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--cif",
        action="append",
        default=[],
        help="mmCIF input, optionally `PATH:CHAIN` to filter to a single chain (repeat for side-by-side)",
    )
    p.add_argument(
        "--pdb-id",
        action="append",
        default=[],
        help="RCSB PDB ID, optionally `ID:CHAIN` to filter to a single chain (repeat for side-by-side)",
    )
    p.add_argument(
        "--dir",
        action="append",
        type=Path,
        default=[],
        help="Directory to scan recursively for model.cif files (sorted; one panel per match)",
    )
    p.add_argument(
        "--confidences",
        action="append",
        default=[],
        help=(
            "Predicted-distogram source: path to a confidences.json (from `deepfold3 inference`). "
            "Reads the per-token expected distance map E[d] directly. "
            "Optionally `PATH:CHAIN` to filter to a single chain via `token_chain_ids`. "
            "Adds one panel per source; in --residual-vs mode shows |E[d] − D_ref| when shapes match."
        ),
    )
    p.add_argument(
        "--sample-dir",
        action="append",
        default=[],
        help=(
            "Inference sample directory containing both `model.cif` and `confidences.json` "
            "(typically `seed-N_sample-M/` from `deepfold3 inference`). Optionally `PATH:CHAIN`. "
            "Accepts shell-style globs (`*`, `?`, `[...]`) — e.g. `runs/foo/step_*/seed-42_sample-0` "
            "expands to all matching dirs (errors if nothing matches). Desugars into one --cif + "
            "one --confidences from each matched dir; control which panels are emitted with --show."
        ),
    )
    p.add_argument(
        "--show",
        choices=("cif", "pred", "both"),
        default="both",
        help="For --sample-dir entries, which panel(s) to emit per dir: cif (coord-derived), pred (E[d]), or both (default).",
    )
    p.add_argument(
        "--residual-vs",
        default=None,
        help=(
            "Reference source (CIF path or RCSB id, optionally `:CHAIN`). When set, every other "
            "source is rendered as |D - D_ref| instead of the raw distance map."
        ),
    )
    p.add_argument(
        "--residual-vmax",
        type=float,
        default=None,
        help="Color-scale cap for residual panels in Å (default: per-figure max).",
    )
    p.add_argument(
        "--residual-cmap",
        default="magma",
        help="Colormap for residual panels (default: magma).",
    )
    p.add_argument(
        "--mae-curve",
        action="store_true",
        help=(
            "With --residual-vs: collapse all residuals into a single MAE-vs-step line plot "
            "(step parsed from `step_NNNN` in the path; falls back to source index)."
        ),
    )
    p.add_argument(
        "--filter",
        default=None,
        help=(
            "Comma-separated kinds to keep: any of {protein,dna,rna,ligand}. "
            "Applied to both source and reference. Default: all."
        ),
    )
    p.add_argument(
        "--align-sequence",
        action="store_true",
        help=(
            "When source and reference shapes mismatch, attempt substring sequence alignment "
            "(no gaps inserted into the query/source) and slice both arrays to the matched span. "
            "Prints a WARNING when alignment is applied. Falls back to current 'show raw' behavior "
            "if no substring containment match exists."
        ),
    )
    p.add_argument(
        "--show-ref",
        action="store_true",
        help=(
            "With --residual-vs: include the native (reference) distogram as the first panel, "
            "embedded into source coordinates when --align-sequence is in effect (NaN-masked at "
            "positions absent from the reference) so all panels share the same row/col indexing."
        ),
    )
    p.add_argument("--output", required=True, type=Path, help="Output PNG path")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "deepfold3" / "rcsb",
        help="Directory to cache fetched mmCIF files",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=DistogramConfig.last_break,
        help=f"Color-scale upper bound in Å (default {DistogramConfig.last_break} — distogram last_break)",
    )
    p.add_argument("--cmap", default="viridis_r", help="Matplotlib colormap")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    def _split_chain(spec: str) -> tuple[str, str | None]:
        if ":" in spec:
            base, chain = spec.rsplit(":", 1)
            return base, chain
        return spec, None

    kind_filter: set[str] | None = None
    if args.filter is not None:
        kind_filter = {k.strip().lower() for k in args.filter.split(",") if k.strip()}
        bad = kind_filter - set(_KIND_OPTIONS)
        if bad:
            p.error(f"--filter: unknown kind(s) {sorted(bad)}; choose from {list(_KIND_OPTIONS)}")

    # Desugar --sample-dir into matching --cif / --confidences entries based
    # on --show. Each sample dir is expected to contain both `model.cif` and
    # `confidences.json` (the layout produced by `deepfold3 inference`).
    # Globs (`*`, `?`, `[...]`) are expanded here so that unquoted shell
    # patterns that bash leaves literal on no-match still produce a useful
    # error or — when matches exist — just work.
    sample_dir_cif_paths: set[Path] = set()  # paths originating from --sample-dir (for paired layout)
    sample_dir_conf_paths: set[Path] = set()
    extra_cif: list[str] = []
    extra_conf: list[str] = []
    for spec in args.sample_dir:
        base, chain = _split_chain(spec)
        suffix = f":{chain}" if chain else ""
        if any(ch in base for ch in "*?["):
            matched = sorted(glob.glob(base), key=_natkey)
            if not matched:
                p.error(f"--sample-dir glob matched nothing: {base!r}")
            dirs = [Path(m) for m in matched]
        else:
            dirs = [Path(base)]
        for d in dirs:
            if not d.is_dir():
                p.error(f"--sample-dir not a directory: {d}")
            cif_path = d / "model.cif"
            conf_path = d / "confidences.json"
            if args.show in ("cif", "both"):
                if not cif_path.exists():
                    p.error(f"--sample-dir {d}: missing model.cif")
                extra_cif.append(f"{cif_path}{suffix}")
                sample_dir_cif_paths.add(cif_path.resolve())
            if args.show in ("pred", "both"):
                if not conf_path.exists():
                    p.error(f"--sample-dir {d}: missing confidences.json")
                extra_conf.append(f"{conf_path}{suffix}")
                sample_dir_conf_paths.add(conf_path.resolve())

    # Natural-sort --cif inputs by path so that shell-glob expansions like
    # `step_*/model.cif` (lex order from bash) are reordered to step_2,
    # step_10, step_100. --dir scans are already natural-sorted below.
    cif_specs = sorted(
        (_split_chain(c) for c in args.cif + extra_cif), key=lambda s: _natkey(s[0])
    )
    for d in args.dir:
        if not d.is_dir():
            p.error(f"--dir not a directory: {d}")
        for found in sorted(d.rglob("model.cif"), key=lambda p: _natkey(str(p))):
            cif_specs.append((str(found), None))
    cif_paths = [Path(base) for base, _ in cif_specs]
    conf_specs = sorted(
        (_split_chain(c) for c in args.confidences + extra_conf), key=lambda s: _natkey(s[0])
    )
    conf_paths = [Path(base) for base, _ in conf_specs]

    sources: list[tuple[str, Path, str | None]] = []
    for spec in args.pdb_id:
        pid, chain = _split_chain(spec)
        label = pid.upper() if chain is None else f"{pid.upper()}:{chain}"
        sources.append((label, _fetch_pdb(pid, args.cache_dir), chain))
    common = Path(__import__("os").path.commonpath([str(p) for p in cif_paths])) if len(cif_paths) > 1 else None
    for path, (_, chain) in zip(cif_paths, cif_specs, strict=True):
        if common is not None and common != Path(path.name):
            try:
                label = str(path.relative_to(common))
            except ValueError:
                label = path.name
        else:
            label = path.name
        if chain is not None:
            label = f"{label}:{chain}"
        sources.append((label, path, chain))
    if not sources and not conf_paths:
        p.error("provide at least one of --cif, --pdb-id, --confidences, or --sample-dir")

    # Build labels for confidences sources (mirrors --cif common-path stripping).
    conf_sources: list[tuple[str, Path, str | None]] = []
    conf_common = (
        Path(__import__("os").path.commonpath([str(p) for p in conf_paths])) if len(conf_paths) > 1 else None
    )
    for path, (_, chain) in zip(conf_paths, conf_specs, strict=True):
        if conf_common is not None and conf_common != Path(path.name):
            try:
                label = str(path.relative_to(conf_common))
            except ValueError:
                label = path.name
        else:
            label = path.name
        if chain is not None:
            label = f"{label}:{chain}"
        conf_sources.append((label, path, chain))

    # Optional reference for residual mode
    ref_xyz_full: np.ndarray | None = None
    ref_chains_full: list[str] | None = None
    ref_seq_full: str | None = None
    ref_D_full: np.ndarray | None = None
    ref_label: str | None = None
    if args.residual_vs is not None:
        rspec, rchain = _split_chain(args.residual_vs)
        rpath = Path(rspec)
        if not rpath.exists():
            rpath = _fetch_pdb(rspec, args.cache_dir)
        ref_st = _read_structure(rpath)
        ref_xyz_full, ref_chains_full, _, ref_seq_full = extract_tokens(
            ref_st, chain_filter=rchain, kind_filter=kind_filter
        )
        if not ref_chains_full:
            raise ValueError(f"no tokens found for reference {args.residual_vs!r}")
        ref_D_full = np.linalg.norm(
            ref_xyz_full[:, None, :] - ref_xyz_full[None, :, :], axis=-1
        )
        ref_label = args.residual_vs if rchain is not None else rpath.name
        print(f"[plot_distogram] residual reference {ref_label}: N={len(ref_chains_full)}")

    # Pre-extract all sources so we can pick a common anchor when aligning.
    extracted: list[tuple[str, np.ndarray, list[str], str, Path]] = []  # (label, xyz, chains, seq, path)
    for label, path, chain in sources:
        st = _read_structure(path)
        xyz, chains, _, src_seq = extract_tokens(st, chain_filter=chain, kind_filter=kind_filter)
        if not chains:
            raise ValueError(f"no tokens found for {label} (chain filter: {chain!r})")
        extracted.append((label, xyz, chains, src_seq, path))

    # Anchor for non-residual alignment: the source with the most tokens (no gap inserted into it).
    anchor_idx: int | None = None
    if args.align_sequence and ref_D_full is None and len(extracted) > 1:
        anchor_idx = max(range(len(extracted)), key=lambda i: extracted[i][1].shape[0])
        S_anchor = extracted[anchor_idx][1].shape[0]
        # Only engage anchor mode if at least one source has a different shape
        if all(extracted[i][1].shape[0] == S_anchor for i in range(len(extracted))):
            anchor_idx = None

    panels: list[tuple[str, np.ndarray, list[str], bool]] = []  # (title, matrix, chains, is_residual)
    # Parallel metadata: (source_path | None, sub_order, mae | None).
    # sub_order: 0 = cif/structure, 1 = predicted-distogram, -1 = reference panel.
    panel_meta: list[tuple[Path | None, int, float | None]] = []
    # Raw full-token-space matrix per panel (D for cif, E for pred). Used to
    # compute pred-vs-structure MAE in the paired layout regardless of any
    # anchor-alignment reshape applied to the display panel.
    panel_raw: dict[int, np.ndarray] = {}
    ref_panel_pending: tuple[str, np.ndarray, list[str], bool] | None = None
    # Per parent-dir substring alignment (si, ri, L) computed by the cif loop.
    # Reused by the predicted-distogram loop so pred residuals can be computed
    # even though confidences.json carries no sequence to align on.
    per_dir_align: dict[str, tuple[int, int, int]] = {}

    for i, (label, xyz, chains, src_seq, src_path) in enumerate(extracted):
        # Build the native panel once, in coordinates of the first source (so it shares row/col indexing).
        if args.show_ref and ref_D_full is not None and ref_panel_pending is None:
            S0 = xyz.shape[0]
            if S0 == ref_xyz_full.shape[0]:
                native_panel = ref_D_full.copy()
                ref_chains_panel = ref_chains_full or chains
                ref_title = f"NATIVE {ref_label}\nN={S0}"
            elif args.align_sequence and (al0 := _align_substr(src_seq, ref_seq_full)) is not None:
                s0, r0, L0 = al0
                native_panel = np.full((S0, S0), np.nan, dtype=np.float64)
                native_panel[s0 : s0 + L0, s0 : s0 + L0] = ref_D_full[r0 : r0 + L0, r0 : r0 + L0]
                ref_chains_panel = chains
                ref_title = (
                    f"NATIVE {ref_label}\n"
                    f"src coords  L={L0}/{S0}  trim h/t={s0}/{S0 - s0 - L0}"
                )
            else:
                native_panel = ref_D_full
                ref_chains_panel = ref_chains_full or []
                ref_title = f"NATIVE {ref_label}\nN={ref_xyz_full.shape[0]}"
            ref_panel_pending = (ref_title, native_panel, ref_chains_panel, False)

        if ref_D_full is not None:
            # ---- Residual mode ----
            if xyz.shape[0] == ref_xyz_full.shape[0]:
                D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
                R = np.abs(D - ref_D_full)
                mae = float(R.mean())
                title = f"|{label} − {ref_label}|\nN={len(chains)}  MAE={mae:.2f} Å"
                panels.append((title, R, chains, True))
                panel_meta.append((src_path, 0, mae))
                panel_raw[len(panels) - 1] = D
                print(f"[plot_distogram] {label}: residual MAE={mae:.2f} Å, max={R.max():.2f} Å")
            elif args.align_sequence and (align := _align_substr(src_seq, ref_seq_full)) is not None:
                si, ri, L = align
                S = xyz.shape[0]
                src_drop = (si, S - si - L)
                ref_drop = (ri, ref_xyz_full.shape[0] - ri - L)
                print(
                    f"[plot_distogram] WARNING: {label}: aligning by substring "
                    f"src[{si}:{si + L}] ↔ ref[{ri}:{ri + L}] (L={L}); "
                    f"trimmed src head/tail = {src_drop[0]}/{src_drop[1]}, "
                    f"ref head/tail = {ref_drop[0]}/{ref_drop[1]}"
                )
                D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
                ref_xyz_a = ref_xyz_full[ri : ri + L]
                ref_D_a = np.linalg.norm(ref_xyz_a[:, None, :] - ref_xyz_a[None, :, :], axis=-1)
                R = np.full((S, S), np.nan, dtype=np.float64)
                R[si : si + L, si : si + L] = np.abs(D[si : si + L, si : si + L] - ref_D_a)
                mae = float(np.nanmean(R))
                rmax = float(np.nanmax(R))
                title = (
                    f"|{label} − {ref_label}|\n"
                    f"aligned L={L}/{S}  trim h/t={src_drop[0]}/{src_drop[1]}  MAE={mae:.2f} Å"
                )
                panels.append((title, R, chains, True))
                panel_meta.append((src_path, 0, mae))
                panel_raw[len(panels) - 1] = D
                per_dir_align[str(src_path.parent)] = (si, ri, L)
                print(f"[plot_distogram] {label}: residual MAE={mae:.2f} Å, max={rmax:.2f} Å")
            else:
                D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
                msg = (
                    f"shape {D.shape} vs ref {ref_D_full.shape}; "
                    + ("substring alignment failed" if args.align_sequence else "pass --align-sequence to attempt alignment")
                )
                print(f"[plot_distogram] {label}: skipping residual ({msg}); showing raw distances instead")
                panels.append((f"{label}\nN={len(chains)}", D, chains, False))
                panel_meta.append((src_path, 0, None))
                panel_raw[len(panels) - 1] = D
        else:
            # ---- Raw distogram mode ----
            D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
            anchor_shape = extracted[anchor_idx][1].shape[0] if anchor_idx is not None else None
            if anchor_idx is None or i == anchor_idx or xyz.shape[0] == anchor_shape:
                panels.append((f"{label}\nN={len(chains)}", D, chains, False))
                panel_meta.append((src_path, 0, None))
                panel_raw[len(panels) - 1] = D
                print(f"[plot_distogram] {label}: {len(chains)} tokens, max d = {D.max():.1f} Å")
            else:
                anchor_xyz = extracted[anchor_idx][1]
                anchor_chains = extracted[anchor_idx][2]
                anchor_seq = extracted[anchor_idx][3]
                S = anchor_xyz.shape[0]
                align = _align_substr(anchor_seq, src_seq)
                if align is None:
                    print(
                        f"[plot_distogram] WARNING: {label}: cannot align to anchor "
                        f"({len(anchor_seq)} vs {len(src_seq)}, no substring match); "
                        "panel kept at original size"
                    )
                    panels.append((f"{label}\nN={len(chains)}", D, chains, False))
                    panel_meta.append((src_path, 0, None))
                    panel_raw[len(panels) - 1] = D
                else:
                    ai, si, L = align  # ai=anchor offset, si=src offset
                    print(
                        f"[plot_distogram] WARNING: {label}: aligning to anchor "
                        f"anchor[{ai}:{ai + L}] ↔ src[{si}:{si + L}] (L={L}); "
                        f"src trim h/t={si}/{xyz.shape[0] - si - L}"
                    )
                    D_emb = np.full((S, S), np.nan, dtype=np.float64)
                    D_emb[ai : ai + L, ai : ai + L] = D[si : si + L, si : si + L]
                    title = (
                        f"{label}\n"
                        f"anchor coords  L={L}/{S}  trim h/t={si}/{xyz.shape[0] - si - L}"
                    )
                    panels.append((title, D_emb, anchor_chains, False))
                    panel_meta.append((src_path, 0, None))
                    panel_raw[len(panels) - 1] = D

    if ref_panel_pending is not None:
        panels.insert(0, ref_panel_pending)
        panel_meta.insert(0, (None, -1, None))

    # ---- Predicted distogram panels (from confidences.json) ----
    # E[d] is already in Å (per-token expected distance from softmax over bins).
    # No xyz, no kind filter (json doesn't carry token kind); chain filter via
    # `token_chain_ids` is applied in `_load_predicted_distogram`. In residual
    # mode we pair with the reference D_ref when shapes match; mismatched
    # shapes fall back to showing raw E[d].
    if conf_sources and kind_filter is not None:
        print(
            f"[plot_distogram] WARNING: --filter {sorted(kind_filter)} is ignored for "
            f"--confidences sources (token-kind not stored in confidences.json); apply "
            "via :CHAIN if needed"
        )
    for label, path, chain in conf_sources:
        E, chains = _load_predicted_distogram(path, chain_filter=chain)
        if ref_D_full is not None:
            if E.shape == ref_D_full.shape:
                R = np.abs(E - ref_D_full)
                mae = float(R.mean())
                title = f"|{label} pred − {ref_label}|\nN={len(chains)}  MAE={mae:.2f} Å"
                panels.append((title, R, chains, True))
                panel_meta.append((path, 1, mae))
                panel_raw[len(panels) - 1] = E
                print(f"[plot_distogram] {label} (pred): residual MAE={mae:.2f} Å, max={R.max():.2f} Å")
            elif (align := per_dir_align.get(str(path.parent))) is not None:
                # Inherit substring alignment from the matching cif in the same sample-dir.
                si, ri, L = align
                S = E.shape[0]
                ref_D_a = ref_D_full[ri : ri + L, ri : ri + L]
                R = np.full((S, S), np.nan, dtype=np.float64)
                R[si : si + L, si : si + L] = np.abs(E[si : si + L, si : si + L] - ref_D_a)
                mae = float(np.nanmean(R))
                rmax = float(np.nanmax(R))
                title = (
                    f"|{label} pred − {ref_label}|\n"
                    f"aligned L={L}/{S}  trim h/t={si}/{S - si - L}  MAE={mae:.2f} Å"
                )
                panels.append((title, R, chains, True))
                panel_meta.append((path, 1, mae))
                panel_raw[len(panels) - 1] = E
                print(f"[plot_distogram] {label} (pred): residual MAE={mae:.2f} Å, max={rmax:.2f} Å (cif-borrowed alignment)")
            else:
                print(
                    f"[plot_distogram] {label} (pred): shape {E.shape} vs ref {ref_D_full.shape}; "
                    "showing raw E[d] instead"
                )
                panels.append((f"{label} pred E[d]\nN={len(chains)}", E, chains, False))
                panel_meta.append((path, 1, None))
                panel_raw[len(panels) - 1] = E
        else:
            panels.append((f"{label} pred E[d]\nN={len(chains)}", E, chains, False))
            panel_meta.append((path, 1, None))
            panel_raw[len(panels) - 1] = E
            print(f"[plot_distogram] {label} (pred): {len(chains)} tokens, max E[d] = {E.max():.1f} Å")

    # MAE-vs-step single panel
    if args.mae_curve:
        if ref_D_full is None:
            p.error("--mae-curve requires --residual-vs")

        residual_panels = [(title, m) for title, m, _, is_r in panels if is_r]
        if not residual_panels:
            p.error("--mae-curve: no residual panels were produced (shape mismatch with reference?)")

        step_re = re.compile(r"step_(\d+)")
        xs: list[float] = []
        ys: list[float] = []
        for idx, (title, M) in enumerate(residual_panels):
            match = step_re.search(title)
            xs.append(float(match.group(1)) if match else float(idx))
            ys.append(float(np.nanmean(M)))

        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, marker="o", color="tab:blue")
        ax.set_xlabel("training step" if step_re.search(residual_panels[0][0]) else "source index")
        ax.set_ylabel("MAE  |D − D_ref|  (Å)")
        ax.set_title(f"Distogram MAE vs {ref_label}")
        ax.grid(True, alpha=0.3)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"[plot_distogram] saved {args.output}")
        return

    # Shared residual vmax (so all residual panels are comparable)
    residual_vmax = args.residual_vmax
    if ref_D_full is not None and residual_vmax is None:
        res_max = max((float(np.nanmax(m)) for _, m, _, is_r in panels if is_r), default=1.0)
        residual_vmax = res_max if res_max > 0 else 1.0

    # ---- Paired sample-dir layout (--sample-dir + --show=both) ----
    # Pairs (cif | pred) per sample-dir, packed onto rows. Column count is
    # picked to make the figure as close to square as possible while keeping
    # `cols` even so that every (cif, pred) pair lands in the same row.
    # Header row hosts the native reference (when --show-ref + --residual-vs)
    # and a MAE-vs-step line plot — drawn whenever any per-step MAE exists,
    # whether against the ref (residual mode) or pred-vs-structure
    # (no-ref mode). Falls through to the auto-square-grid path otherwise so
    # ad-hoc --cif/--pdb-id callers keep their existing layout.
    paired_mode = bool(args.sample_dir) and args.show == "both"
    if paired_mode:
        groups: dict[str, dict[int, int]] = {}
        ref_idx: int | None = None
        ungrouped: list[int] = []
        for i, (src, sub, _mae) in enumerate(panel_meta):
            if sub == -1:
                ref_idx = i
                continue
            if src is None:
                ungrouped.append(i)
                continue
            parent = str(src.parent)
            groups.setdefault(parent, {})[sub] = i
        # Demote groups missing a partner (only cif or only pred) to the
        # ungrouped pool — they shouldn't take up a body row reserved for
        # cif/pred pairs. Keeps `sorted_parents` to fully-paired groups.
        complete_parents: list[str] = []
        for parent, g in groups.items():
            if 0 in g and 1 in g:
                complete_parents.append(parent)
            else:
                ungrouped.extend(g.values())
        sorted_parents = sorted(complete_parents, key=_natkey)
        step_re = re.compile(r"step_(\d+)")

        # Build the MAE-vs-step series.
        xs_step: list[float] = []
        ys_struct_vs_ref: list[float] = []
        ys_pred_vs_ref: list[float] = []
        ys_pred_vs_struct: list[float] = []
        for parent in sorted_parents:
            cif_idx = groups[parent].get(0)
            pred_idx = groups[parent].get(1)
            cif_mae = panel_meta[cif_idx][2] if cif_idx is not None else None
            pred_mae = panel_meta[pred_idx][2] if pred_idx is not None else None
            pvs_mae: float | None = None
            if cif_idx is not None and pred_idx is not None:
                D = panel_raw.get(cif_idx)
                E = panel_raw.get(pred_idx)
                if D is not None and E is not None and D.shape == E.shape:
                    pvs_mae = float(np.abs(E - D).mean())
            if cif_mae is None and pred_mae is None and pvs_mae is None:
                continue
            m = step_re.search(parent)
            xs_step.append(float(m.group(1)) if m else float(len(xs_step)))
            ys_struct_vs_ref.append(cif_mae if cif_mae is not None else float("nan"))
            ys_pred_vs_ref.append(pred_mae if pred_mae is not None else float("nan"))
            ys_pred_vs_struct.append(pvs_mae if pvs_mae is not None else float("nan"))

        has_mae_data = bool(xs_step) and any(
            any(v == v for v in series)  # non-NaN
            for series in (ys_struct_vs_ref, ys_pred_vs_ref, ys_pred_vs_struct)
        )

        def _make_image_cell(idx: int) -> dict:
            return {"kind": "image", "panel": panels[idx]}

        # Header row: native + MAE side by side. The "native" cell is the
        # reference panel when --residual-vs+--show-ref are set; when absent
        # we promote a lone ungrouped panel (typically --pdb-id used as the
        # ground-truth source in non-residual mode) so that whatever stands
        # in for the native distogram sits next to the MAE plot.
        native_idx: int | None = None
        if ref_idx is not None:
            native_idx = ref_idx
        elif ungrouped:
            native_idx = ungrouped.pop(0)

        header_cells: list[dict] = []
        if native_idx is not None:
            header_cells.append(_make_image_cell(native_idx))
        if has_mae_data:
            header_cells.append(
                {
                    "kind": "mae",
                    "xs": xs_step,
                    "y_series": [
                        ("structure vs ref", ys_struct_vs_ref, "tab:blue", "o"),
                        ("predicted vs ref", ys_pred_vs_ref, "tab:orange", "s"),
                        ("predicted vs structure", ys_pred_vs_struct, "tab:green", "^"),
                    ],
                    "ref_label": ref_label,
                    "has_ref": ref_D_full is not None,
                }
            )
        has_header_row = bool(header_cells)

        # Choose column count. Body needs `2 * len(sorted_parents)` cells +
        # one cell per ungrouped panel; pairs require even cols. Iterate
        # candidate even cols and pick the one whose figure aspect (cols·4.5
        # by total_rows·4.0) is closest to 1. Single header row is reserved
        # whether or not it spans all columns.
        body_cells = 2 * len(sorted_parents) + len(ungrouped)
        max_cols = max(2, body_cells)
        best_cols = 2
        best_score = float("inf")
        best_total_rows = 1
        for c in range(2, max_cols + 2, 2):
            body_rows = max(1, math.ceil(body_cells / c)) if body_cells else 0
            total_rows = body_rows + (1 if has_header_row else 0)
            if total_rows == 0:
                continue
            aspect = (c * 4.5) / (total_rows * 4.0)
            score = abs(aspect - 1.0)
            if score < best_score:
                best_score = score
                best_cols = c
                best_total_rows = total_rows
        cols = best_cols

        rows_cells: list[list[dict]] = []
        if has_header_row:
            row = list(header_cells)
            while len(row) < cols:
                row.append({"kind": "blank"})
            rows_cells.append(row)

        cur_row: list[dict] = []
        for parent in sorted_parents:
            cif_idx = groups[parent].get(0)
            pred_idx = groups[parent].get(1)
            cur_row.append(
                _make_image_cell(cif_idx) if cif_idx is not None else {"kind": "blank"}
            )
            cur_row.append(
                _make_image_cell(pred_idx) if pred_idx is not None else {"kind": "blank"}
            )
            if len(cur_row) >= cols:
                rows_cells.append(cur_row)
                cur_row = []
        for i in ungrouped:
            cur_row.append(_make_image_cell(i))
            if len(cur_row) >= cols:
                rows_cells.append(cur_row)
                cur_row = []
        if cur_row:
            while len(cur_row) < cols:
                cur_row.append({"kind": "blank"})
            rows_cells.append(cur_row)

        n_rows = len(rows_cells)
        fig, axes = plt.subplots(
            n_rows, cols, figsize=(4.5 * cols, 4.0 * n_rows), squeeze=False, layout="constrained"
        )
        for ri, row in enumerate(rows_cells):
            for ci, cell in enumerate(row):
                ax = axes[ri][ci]
                if cell["kind"] == "blank":
                    ax.axis("off")
                elif cell["kind"] == "image":
                    title, M, chains, is_residual = cell["panel"]
                    if is_residual:
                        plot_panel(ax, M, chains, title, vmax=residual_vmax, cmap=args.residual_cmap)
                    else:
                        plot_panel(ax, M, chains, title, vmax=args.vmax, cmap=args.cmap)
                elif cell["kind"] == "mae":
                    drawn_any = False
                    for name, ys, color, marker in cell["y_series"]:
                        if any(v == v for v in ys):
                            ax.plot(cell["xs"], ys, marker=marker, color=color, label=name)
                            drawn_any = True
                    ax.set_xlabel("training step")
                    ax.set_ylabel("MAE (Å)")
                    ax.set_title(
                        f"Distogram MAE vs {cell['ref_label']}"
                        if cell["has_ref"]
                        else "Distogram MAE (predicted vs structure)"
                    )
                    if drawn_any:
                        ax.legend(fontsize=8, loc="best")
                    ax.grid(True, alpha=0.3)
        if ref_D_full is not None:
            fig.suptitle(
                f"Residual |D − D_ref| vs {ref_label}  (cap {residual_vmax:.1f} Å, {args.residual_cmap}); "
                "pairs: cif | predicted",
                fontsize=11,
            )
        else:
            fig.suptitle(
                f"Distogram (clipped at {args.vmax:.1f} Å); pairs: cif | predicted",
                fontsize=11,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"[plot_distogram] saved {args.output}")
        return

    n = len(panels)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False, layout="constrained"
    )
    flat_axes = axes.ravel()
    for ax, (title, M, chains, is_residual) in zip(flat_axes, panels, strict=False):
        if is_residual:
            plot_panel(ax, M, chains, title, vmax=residual_vmax, cmap=args.residual_cmap)
        else:
            plot_panel(ax, M, chains, title, vmax=args.vmax, cmap=args.cmap)
    for ax in flat_axes[n:]:
        ax.axis("off")

    if ref_D_full is not None:
        fig.suptitle(
            f"Residual |D − D_ref| vs {ref_label}  (cap {residual_vmax:.1f} Å, {args.residual_cmap})", fontsize=11
        )
    else:
        fig.suptitle(f"Distogram — pairwise rep-atom distances (clipped at {args.vmax:.1f} Å)", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot_distogram] saved {args.output}")


if __name__ == "__main__":
    main()
