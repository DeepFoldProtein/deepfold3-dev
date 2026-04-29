"""Tests for canonical-sequence helpers and process_mmcif msa_id derivation."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

PATCHR_CIF_DIR = Path(__file__).parent.parent / "data" / "patchr" / "cifs"


def _gemmi_required():
    try:
        import gemmi  # noqa: F401
    except ImportError:
        pytest.skip("gemmi not installed")


def _patchr_cifs_required():
    if not PATCHR_CIF_DIR.exists():
        pytest.skip(f"PATCHR cif dir not found: {PATCHR_CIF_DIR}")


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------


def test_seqres_one_letter_microhet():
    """Microheterogeneity (`MSE,MET`) collapses to first alternative.

    Note: gemmi distinguishes modified (lowercase) vs standard (uppercase)
    in its raw ``_seqres_one_letter`` output. ``get_canonical_one_letter``
    is what uppercases for MSA search; this test validates the lower-level
    helper preserves the case-sensitivity gemmi provides.
    """
    _gemmi_required()
    from deepfold3.data.prepare.canonical import _seqres_one_letter

    assert _seqres_one_letter("MET") == "M"
    # MSE → selenomethionine, parent MET → 'm' (lowercase = modified, gemmi convention)
    assert _seqres_one_letter("MSE") == "m"
    # microhet entry — first alternative wins
    assert _seqres_one_letter("MSE,MET") == "m"
    # Unknown / placeholder → X
    assert _seqres_one_letter("???") == "X"
    assert _seqres_one_letter("") == "X"


def test_get_canonical_one_letter_uppercase():
    """``get_canonical_one_letter`` always returns uppercase (MSA-search ready)."""
    _gemmi_required()
    _patchr_cifs_required()
    import gemmi

    from deepfold3.data.prepare.canonical import get_canonical_one_letter

    path = PATCHR_CIF_DIR / "10gs.cif"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    saw_polymer = False
    for entity in st.entities:
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        seq = get_canonical_one_letter(entity)
        if seq is None:
            continue
        saw_polymer = True
        assert seq == seq.upper(), f"non-uppercase letters in canonical seq: {seq!r}"
        assert all(c.isalpha() or c == "X" for c in seq), f"non-alpha in canonical: {seq!r}"
    assert saw_polymer


def test_seqres_three_letter_nonstandard_mapped():
    """3-letter form applies NONSTANDARD_TO_STANDARD; matches `_chain_sequence` behavior."""
    from deepfold3.data import const
    from deepfold3.data.prepare.canonical import _seqres_three_letter

    # Standard residue passes through
    assert _seqres_three_letter("ALA") == "ALA"
    assert _seqres_three_letter("GLY") == "GLY"
    # Modified residue → standard parent
    assert _seqres_three_letter("MSE") == "MET"
    assert _seqres_three_letter("SEP") == "SER"
    # Microhet → first alternative, then mapping
    assert _seqres_three_letter("MSE,MET") == "MET"
    # Unmapped non-standard stays as-is (later turned into UNK by the loss)
    unknown = "ZZZ"
    assert unknown not in const._NONSTANDARD_TO_STANDARD
    assert _seqres_three_letter(unknown) == "ZZZ"
    # Empty → UNK
    assert _seqres_three_letter("") == "UNK"


def test_get_canonical_three_letter_none_for_no_full_sequence():
    """Returns None when entity has no full_sequence (graceful for ligands / pathological inputs)."""
    from deepfold3.data.prepare.canonical import get_canonical_three_letter

    class _FakeEntity:
        full_sequence = []

    assert get_canonical_three_letter(_FakeEntity()) is None
    assert get_canonical_three_letter(None) is None


# ---------------------------------------------------------------------------
# End-to-end via process_mmcif on real PATCHR CIFs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cif_name",
    [
        "10gs.cif",  # 2 protein chains + ligand
        "103l.cif",  # single chain T4 lysozyme
        "108l.cif",  # single chain
    ],
)
def test_process_mmcif_populates_canonical_seqs(cif_name):
    _gemmi_required()
    _patchr_cifs_required()
    from deepfold3.data.prepare.structure import process_mmcif

    path = PATCHR_CIF_DIR / cif_name
    if not path.exists():
        pytest.skip(f"fixture {path} missing")

    structure, record = process_mmcif(path)

    # Every protein chain must have a canonical sequence string.
    polymer_chains = [c for c in record.chains if c.mol_type == 0]
    assert polymer_chains, "expected at least one polymer chain"
    canonical_seqs = structure.chain_canonical_seqs or {}
    for ci in polymer_chains:
        seq = canonical_seqs.get(int(ci.chain_id))
        assert seq is not None, f"missing canonical for asym_id={ci.chain_id}"
        assert seq, "canonical sequence should be non-empty"
        assert all(c.isalpha() for c in seq), f"non-letter in canonical: {seq!r}"


def test_process_mmcif_msa_id_matches_canonical_three_letter_hash():
    """The msa_id stored on ChainInfo equals SHA256("|".join(canonical 3-letter mon_ids))."""
    _gemmi_required()
    _patchr_cifs_required()
    import gemmi

    from deepfold3.data.prepare.canonical import get_canonical_three_letter
    from deepfold3.data.prepare.structure import process_mmcif

    path = PATCHR_CIF_DIR / "103l.cif"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")

    structure, record = process_mmcif(path)

    # Re-read the source to get entity → canonical 3-letter list independently.
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    name_to_three = {}
    for entity in st.entities:
        if entity.entity_type != gemmi.EntityType.Polymer:
            continue
        three = get_canonical_three_letter(entity)
        if three is not None:
            name_to_three[entity.name] = three

    # For each polymer chain, the msa_id must match the canonical hash.
    for ci in record.chains:
        if ci.mol_type != 0:
            continue
        # entity_id is our internal numeric id, not gemmi entity name; instead use
        # the chain's canonical sequence (one-letter) which was derived from the
        # same entity and re-derive the 3-letter list from the gemmi side that
        # produced an identical 1-letter string.
        canonical_one = (structure.chain_canonical_seqs or {}).get(int(ci.chain_id))
        assert canonical_one is not None
        # Find the gemmi entity whose canonical 1-letter matches.
        from deepfold3.data.prepare.canonical import get_canonical_one_letter

        match_three: list[str] | None = None
        for entity in st.entities:
            if entity.entity_type != gemmi.EntityType.Polymer:
                continue
            if get_canonical_one_letter(entity) == canonical_one:
                match_three = name_to_three.get(entity.name)
                break
        assert match_three is not None, "could not locate matching entity"

        expected_hash = hashlib.sha256("|".join(match_three).encode()).hexdigest()
        assert str(ci.msa_id) == expected_hash, f"msa_id mismatch on asym_id={ci.chain_id}"


def test_structure_round_trip_preserves_canonical_seqs():
    _gemmi_required()
    _patchr_cifs_required()
    from deepfold3.data.prepare.structure import process_mmcif
    from deepfold3.data.types import Structure

    path = PATCHR_CIF_DIR / "10gs.cif"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")

    st_in, _ = process_mmcif(path)
    assert st_in.chain_canonical_seqs

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "10gs.npz"
        st_in.save(out)
        st_out = Structure.load(out)

    assert st_out.chain_canonical_seqs == st_in.chain_canonical_seqs


def test_structure_load_backward_compat_no_canonical_field():
    """Old NPZs without the canonical field still load; chain_canonical_seqs is None."""
    import numpy as np

    from deepfold3.data.types import ATOM_DTYPE, BOND_DTYPE, CHAIN_DTYPE, RESIDUE_DTYPE, Structure

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "old.npz"
        np.savez_compressed(
            str(out),
            atoms=np.empty(0, dtype=ATOM_DTYPE),
            bonds=np.empty(0, dtype=BOND_DTYPE),
            residues=np.empty(0, dtype=RESIDUE_DTYPE),
            chains=np.empty(0, dtype=CHAIN_DTYPE),
            name=np.array("legacy"),
        )
        st = Structure.load(out)
    assert st.chain_canonical_seqs is None
    assert st.name == "legacy"


def test_extract_sequences_uses_canonical_cache():
    _gemmi_required()
    _patchr_cifs_required()
    from deepfold3.data.paths import structure_path
    from deepfold3.data.prepare.sequences import extract_sequences_by_msa_id
    from deepfold3.data.prepare.structure import process_mmcif

    cifs = ["10gs.cif", "103l.cif", "108l.cif"]
    paths = [PATCHR_CIF_DIR / c for c in cifs if (PATCHR_CIF_DIR / c).exists()]
    if not paths:
        pytest.skip("no PATCHR fixtures available")

    with tempfile.TemporaryDirectory() as tmp:
        structures_dir = Path(tmp) / "structures"
        records = []
        for p in paths:
            st, rec = process_mmcif(p)
            sp = structure_path(structures_dir, rec.id)
            sp.parent.mkdir(parents=True, exist_ok=True)
            st.save(sp)
            records.append(rec)

        seqs = extract_sequences_by_msa_id(records, structures_dir)

    # The returned seq for each msa_id should match the canonical seq stored
    # on the originating chain.
    assert seqs
    for record, path in zip(records, paths, strict=True):
        st, _ = process_mmcif(path)
        canonical = st.chain_canonical_seqs or {}
        for ci in record.chains:
            if ci.mol_type != 0:
                continue
            mid = str(ci.msa_id)
            if mid in seqs:
                assert seqs[mid] == canonical.get(int(ci.chain_id))
