"""MMseqs2-based MSA search — server API and local binary.

Supports both the ColabFold public server (``search_server``) and a local
``mmseqs`` installation (``search_local``).  Both return an ``MsaResult``
whose A3M text can be fed directly to ``parse_a3m()``.

Protocol details reimplemented from scratch based on the ColabFold project.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MsaResult:
    """Result of an MSA search.

    Attributes:
        unpaired: Per-chain unpaired A3M text, keyed by chain index (0-based).
        paired:   Per-chain paired A3M text (empty dict for monomer).
        m8_text:  Raw ``pdb70.m8`` text from template search (empty if not available).
    """

    unpaired: dict[int, str]
    paired: dict[int, str] = dataclasses.field(default_factory=dict)
    m8_text: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_query_fasta(sequences: list[str]) -> str:
    """Format sequences as FASTA with ``>101``, ``>102``, ... headers."""
    parts: list[str] = []
    for i, seq in enumerate(sequences):
        parts.append(f">{101 + i}\n{seq}\n")
    return "".join(parts)


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse simple FASTA text into ``(name, sequence)`` pairs."""
    entries: list[tuple[str, str]] = []
    name: str | None = None
    seq_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                entries.append((name, "".join(seq_parts)))
            name = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)
    if name is not None:
        entries.append((name, "".join(seq_parts)))
    return entries


def _sequence_hash(sequences: list[str]) -> str:
    """SHA-256 of sorted sequences, truncated to 16 hex chars."""
    h = hashlib.sha256("\n".join(sorted(sequences)).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Template hit parsing (pdb70.m8)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TemplateM8Hit:
    """A single template hit from pdb70.m8 (BLAST tabular format)."""

    pdb_id: str
    chain_id: str
    identity: float
    evalue: float
    qstart: int
    qend: int
    tstart: int
    tend: int


def parse_m8(m8_text: str) -> dict[int, list[TemplateM8Hit]]:
    """Parse pdb70.m8 text into per-chain template hits.

    Returns:
        Dict mapping chain index (0-based) to list of TemplateM8Hit.
    """
    hits: dict[int, list[TemplateM8Hit]] = {}
    for line in m8_text.strip().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        try:
            chain_idx = int(parts[0]) - 101
            target = parts[1]
            identity = float(parts[2])
            qstart = int(parts[6])
            qend = int(parts[7])
            tstart = int(parts[8])
            tend = int(parts[9])
            evalue = float(parts[10])
        except (ValueError, IndexError):
            continue

        # Parse PDB ID + chain from target (e.g. "1abc_A")
        if "_" in target:
            pdb_id = target[:4].upper()
            chain_id_str = target[5:]
        else:
            pdb_id = target[:4].upper()
            chain_id_str = "A"

        if chain_idx not in hits:
            hits[chain_idx] = []
        hits[chain_idx].append(
            TemplateM8Hit(
                pdb_id=pdb_id,
                chain_id=chain_id_str,
                identity=identity,
                evalue=evalue,
                qstart=qstart,
                qend=qend,
                tstart=tstart,
                tend=tend,
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(sequences: list[str], use_env: bool, use_pairing: bool) -> str:
    """Build a cache sub-directory name incorporating search parameters."""
    base = _sequence_hash(sequences)
    suffix = ""
    if use_env:
        suffix += "_env"
    if use_pairing:
        suffix += "_paired"
    return base + suffix


def _try_load_cache(cache_dir: Path, key: str, n_chains: int) -> MsaResult | None:
    """Load cached MsaResult, or return None on miss."""
    d = cache_dir / key
    if not d.is_dir():
        return None
    unpaired: dict[int, str] = {}
    paired: dict[int, str] = {}
    for i in range(n_chains):
        uf = d / f"unpaired_{i}.a3m"
        if not uf.is_file():
            return None
        unpaired[i] = uf.read_text()
        pf = d / f"paired_{i}.a3m"
        if pf.is_file():
            paired[i] = pf.read_text()
    logger.info("Loaded MSA from cache: %s", d)
    return MsaResult(unpaired=unpaired, paired=paired)


def _save_cache(cache_dir: Path, key: str, result: MsaResult) -> None:
    """Write MsaResult to cache."""
    d = cache_dir / key
    d.mkdir(parents=True, exist_ok=True)
    for i, text in result.unpaired.items():
        (d / f"unpaired_{i}.a3m").write_text(text)
    for i, text in result.paired.items():
        (d / f"paired_{i}.a3m").write_text(text)
    logger.info("Saved MSA to cache: %s", d)


# ---------------------------------------------------------------------------
# Server API
# ---------------------------------------------------------------------------

_DEFAULT_SERVER = "https://api.colabfold.com"
_CONNECT_TIMEOUT = 6.02  # slightly > multiple of 3, per requests best practice
_MAX_RETRIES = 5


def _urlopen_with_retry(req: urllib.request.Request, *, timeout: float, max_retries: int = _MAX_RETRIES) -> bytes:
    """Open a URL request with retries on transient errors."""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt >= max_retries - 1:
                raise
            logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(5)
    raise RuntimeError("unreachable")  # pragma: no cover


def _submit(url: str, query_fasta: str, mode: str, headers: dict[str, str]) -> dict:
    """POST to the submission endpoint, return JSON response."""
    endpoint = "ticket/pair" if mode.startswith("pair") else "ticket/msa"
    data = urllib.parse.urlencode({"q": query_fasta, "mode": mode}).encode()
    req = urllib.request.Request(f"{url}/{endpoint}", data=data, headers=headers, method="POST")
    body = _urlopen_with_retry(req, timeout=_CONNECT_TIMEOUT)
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        logger.error("Server didn't reply with JSON: %s", body[:200])
        return {"status": "ERROR"}


def _poll(url: str, ticket_id: str, headers: dict[str, str], timeout: float) -> None:
    """Poll until the job reaches COMPLETE or raises on ERROR/timeout."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"MSA server job {ticket_id} timed out after {timeout}s")
        sleep_time = 5 + random.randint(0, 5)
        time.sleep(sleep_time)
        req = urllib.request.Request(f"{url}/ticket/{ticket_id}", headers=headers)
        body = _urlopen_with_retry(req, timeout=_CONNECT_TIMEOUT)
        try:
            out = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            logger.error("Server didn't reply with JSON: %s", body[:200])
            continue
        status = out.get("status", "UNKNOWN")
        logger.info("Job %s status: %s", ticket_id, status)
        if status == "COMPLETE":
            return
        if status == "ERROR":
            raise RuntimeError(f"MSA server returned ERROR for job {ticket_id}")
        if status == "MAINTENANCE":
            raise RuntimeError("MSA server is undergoing maintenance")
        # RUNNING, PENDING, UNKNOWN, RATELIMIT → keep polling


def _download(url: str, ticket_id: str, headers: dict[str, str]) -> bytes:
    """Download result tar.gz as bytes."""
    req = urllib.request.Request(f"{url}/result/download/{ticket_id}", headers=headers)
    return _urlopen_with_retry(req, timeout=_CONNECT_TIMEOUT)


def _extract_a3m(tar_bytes: bytes, n_chains: int, use_env: bool, use_pairing: bool) -> MsaResult:
    """Extract A3M text from the result tar.gz.

    Files in the archive:
    - ``uniref.a3m`` — unpaired UniRef hits
    - ``bfd.mgnify30.metaeuk30.smag30.a3m`` — environmental hits (if use_env)
    - ``pair.a3m`` — paired hits (if use_pairing)

    Within each file, per-query blocks are separated by ``\\x00`` and keyed
    by the ``>N`` header (N = 101, 102, ...).
    """

    def _split_a3m_blocks(text: str, n: int) -> dict[int, str]:
        """Split A3M text on \\x00 into per-query blocks keyed by chain index."""
        blocks: dict[int, str] = {}
        current_key: int | None = None
        current_lines: list[str] = []
        update_key = True

        for line in text.splitlines(keepends=True):
            if "\x00" in line:
                line = line.replace("\x00", "")
                update_key = True
            if line.startswith(">") and update_key:
                # Save previous block
                if current_key is not None:
                    blocks[current_key] = "".join(current_lines)
                try:
                    header_num = int(line[1:].strip().split()[0])
                    current_key = header_num - 101
                except ValueError:
                    current_key = current_key if current_key is not None else 0
                current_lines = [line]
                update_key = False
            else:
                current_lines.append(line)

        if current_key is not None:
            blocks[current_key] = "".join(current_lines)
        return blocks

    # Extract tar.gz in memory
    file_contents: dict[str, str] = {}
    m8_text = ""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            basename = os.path.basename(member.name)
            if basename.endswith(".a3m"):
                f = tar.extractfile(member)
                if f is not None:
                    file_contents[basename] = f.read().decode("utf-8", errors="replace")
            elif basename == "pdb70.m8":
                f = tar.extractfile(member)
                if f is not None:
                    m8_text = f.read().decode("utf-8", errors="replace")

    unpaired: dict[int, str] = {}
    paired: dict[int, str] = {}

    # Unpaired: uniref + optional env
    uniref_text = file_contents.get("uniref.a3m", "")
    if uniref_text:
        uniref_blocks = _split_a3m_blocks(uniref_text, n_chains)
    else:
        uniref_blocks = {}

    env_text = file_contents.get("bfd.mgnify30.metaeuk30.smag30.a3m", "")
    if use_env and env_text:
        env_blocks = _split_a3m_blocks(env_text, n_chains)
    else:
        env_blocks = {}

    for i in range(n_chains):
        parts = []
        if i in uniref_blocks:
            parts.append(uniref_blocks[i])
        if i in env_blocks:
            # Append env sequences (skip the query header line which duplicates)
            env_lines = env_blocks[i].splitlines(keepends=True)
            # Skip the first header + sequence (query row) since it's already in uniref
            in_query = True
            for line in env_lines:
                if line.startswith(">") and in_query:
                    in_query = False
                    continue
                if in_query:
                    continue
                parts.append(line)
        unpaired[i] = "".join(parts) if parts else ""

    # Paired
    if use_pairing:
        pair_text = file_contents.get("pair.a3m", "")
        if pair_text:
            pair_blocks = _split_a3m_blocks(pair_text, n_chains)
            for i in range(n_chains):
                paired[i] = pair_blocks.get(i, "")

    return MsaResult(unpaired=unpaired, paired=paired, m8_text=m8_text)


def search_server(
    sequences: list[str],
    *,
    server_url: str = _DEFAULT_SERVER,
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    timeout: float = 600,
    cache_dir: Path | None = None,
) -> MsaResult:
    """Search for MSAs using the ColabFold MMseqs2 server API.

    Args:
        sequences: List of amino acid sequences (one per chain).
        server_url: ColabFold API base URL.
        use_env: Include environmental database hits.
        use_filter: Apply server-side MSA filtering.
        use_pairing: Enable cross-chain MSA pairing (multimer).
        pairing_strategy: ``"greedy"`` or ``"complete"`` pairing.
        timeout: Maximum time to wait for server results (seconds).
        cache_dir: Optional directory for caching results.

    Returns:
        MsaResult with per-chain A3M text.
    """
    n_chains = len(sequences)

    # Check cache
    if cache_dir is not None:
        key = _cache_key(sequences, use_env, use_pairing)
        cached = _try_load_cache(cache_dir, key, n_chains)
        if cached is not None:
            return cached

    headers = {"User-Agent": "deepfold3/1.0"}
    query_fasta = _build_query_fasta(sequences)

    # Determine mode strings
    if use_pairing:
        if pairing_strategy == "complete":
            pair_mode = "paircomplete"
        else:
            pair_mode = "pairgreedy"
        if use_env:
            pair_mode += "-env"
    if use_filter:
        unpaired_mode = "env" if use_env else "all"
    else:
        unpaired_mode = "env-nofilter" if use_env else "nofilter"

    # --- Unpaired search ---
    logger.info("Submitting unpaired MSA search (%d chains, mode=%s)", n_chains, unpaired_mode)
    out = _submit(server_url, query_fasta, unpaired_mode, headers)
    while out.get("status") in ("UNKNOWN", "RATELIMIT"):
        sleep_time = 5 + random.randint(0, 5)
        logger.warning("Server status %s, retrying in %ds", out["status"], sleep_time)
        time.sleep(sleep_time)
        out = _submit(server_url, query_fasta, unpaired_mode, headers)
    if out.get("status") == "ERROR":
        raise RuntimeError("MSA server returned ERROR. Check input sequences.")
    if out.get("status") == "MAINTENANCE":
        raise RuntimeError("MSA server is undergoing maintenance.")

    ticket_id = out["id"]
    logger.info("Unpaired job submitted: %s", ticket_id)
    if out.get("status") != "COMPLETE":
        _poll(server_url, ticket_id, headers, timeout)
    tar_bytes = _download(server_url, ticket_id, headers)
    result = _extract_a3m(tar_bytes, n_chains, use_env, use_pairing=False)

    # --- Paired search (if requested) ---
    if use_pairing and n_chains > 1:
        logger.info("Submitting paired MSA search (mode=%s)", pair_mode)
        out = _submit(server_url, query_fasta, pair_mode, headers)
        while out.get("status") in ("UNKNOWN", "RATELIMIT"):
            sleep_time = 5 + random.randint(0, 5)
            logger.warning("Server status %s, retrying in %ds", out["status"], sleep_time)
            time.sleep(sleep_time)
            out = _submit(server_url, query_fasta, pair_mode, headers)
        if out.get("status") == "ERROR":
            raise RuntimeError("MSA server returned ERROR for pairing search.")

        pair_ticket_id = out["id"]
        logger.info("Paired job submitted: %s", pair_ticket_id)
        if out.get("status") != "COMPLETE":
            _poll(server_url, pair_ticket_id, headers, timeout)
        pair_tar_bytes = _download(server_url, pair_ticket_id, headers)
        pair_result = _extract_a3m(pair_tar_bytes, n_chains, use_env=False, use_pairing=True)
        result.paired = pair_result.paired

    # Save cache
    if cache_dir is not None:
        _save_cache(cache_dir, key, result)

    return result


# ---------------------------------------------------------------------------
# Local MMseqs2 binary
# ---------------------------------------------------------------------------


def _run_mmseqs(binary: str, args: list[str | Path]) -> None:
    """Run an mmseqs subcommand, raising on failure."""
    cmd = [binary] + [str(a) for a in args]
    logger.info("Running: %s", " ".join(cmd))
    env = os.environ.copy()
    env["MMSEQS_CALL_DEPTH"] = "1"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("mmseqs stderr: %s", result.stderr)
        raise RuntimeError(f"mmseqs {args[0]} failed with exit code {result.returncode}")


def _search_monomer(
    binary: str,
    db_dir: Path,
    work_dir: Path,
    use_env: bool,
    use_templates: bool,
    filter: bool,
    threads: int,
    gpu: bool,
    gpu_server: bool,
) -> None:
    """Run the monomer unpaired search pipeline.

    Args:
        gpu: Use GPU acceleration (requires MMseqs2 with GPU support).
        gpu_server: Use dedicated GPU server for reduced per-request overhead.
    """
    uniref_db = "uniref30_2302_db"
    env_db = "colabfold_envdb_202108_db"
    template_db = "pdb100_230517"
    qdb = work_dir / "qdb"

    # Check index availability — determines db_load_mode and suffix convention
    db_load_mode = "2"
    db_suffix_seq = ".idx"  # suffix for sequence DB in expandaln
    db_suffix_aln = ".idx"  # suffix for alignment DB in expandaln
    db_suffix_tpl = ".idx"  # suffix for template DB in convertalis
    if not (db_dir / f"{uniref_db}.idx").is_file() and not (db_dir / f"{uniref_db}.idx.index").is_file():
        db_load_mode = "0"
        db_suffix_seq = "_seq"
        db_suffix_aln = "_aln"
        db_suffix_tpl = ""

    if filter:
        align_eval, qsc, max_accept = "10", "0.8", "100000"
    else:
        align_eval, qsc, max_accept = "10", "-20.0", "1000000"

    # GPU ungapped prefilter uses a different prioritization method than CPU k-mer prefilter,
    # which can miss hits at default --max-seqs 10000. Increase to 100000 to recover
    # CPU-equivalent results (MMseqs2 #995/#999).
    max_seqs = "100000" if gpu else "10000"
    search_param: list[str | Path] = [
        "--num-iterations", "3", "--db-load-mode", db_load_mode, "-a", "-e", "0.1", "--max-seqs", max_seqs,
    ]
    template_search_param: list[str | Path] = []
    if gpu:
        search_param += ["--gpu", "1", "--prefilter-mode", "1"]
        template_search_param += ["--gpu", "1", "--prefilter-mode", "1"]
    else:
        # Use direct k-mer threshold instead of -s sensitivity to match ColabFold server.
        # Note: no quotes around the value — ColabFold #804 shows quoted form is silently ignored.
        search_param += ["--prefilter-mode", "0", "--k-score", "seq:96,prof:80"]
        template_search_param += ["-s", "7.5", "--prefilter-mode", "0"]
    if gpu_server:
        search_param += ["--gpu-server", "1"]

    filter_param: list[str | Path] = [
        "--filter-msa", str(int(filter)), "--filter-min-enable", "1000", "--diff", "3000",
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0", "--qsc", "0", "--max-seq-id", "0.95",
    ]
    expand_param: list[str | Path] = [
        "--expansion-mode", "0", "-e", "inf", "--expand-filter-clusters", str(int(filter)), "--max-seq-id", "0.95",
    ]

    # 1. Search UniRef
    _run_mmseqs(
        binary,
        ["search", qdb, db_dir / uniref_db, work_dir / "res", work_dir / "tmp", "--threads", str(threads)]
        + search_param,
    )
    # 2. Extract profile
    _run_mmseqs(binary, ["mvdb", work_dir / "tmp/latest/profile_1", work_dir / "prof_res"])
    # 3. Link headers
    _run_mmseqs(binary, ["lndb", work_dir / "qdb_h", work_dir / "prof_res_h"])
    # 4. Expand alignments
    _run_mmseqs(
        binary,
        [
            "expandaln", qdb, db_dir / f"{uniref_db}{db_suffix_seq}",
            work_dir / "res", db_dir / f"{uniref_db}{db_suffix_aln}",
            work_dir / "res_exp",
            "--db-load-mode", db_load_mode, "--threads", str(threads),
        ]
        + expand_param,
    )
    # 5. Realign
    _run_mmseqs(
        binary,
        [
            "align", work_dir / "prof_res", db_dir / f"{uniref_db}{db_suffix_seq}",
            work_dir / "res_exp", work_dir / "res_exp_realign",
            "--db-load-mode", db_load_mode, "-e", align_eval, "--max-accept", max_accept,
            "--threads", str(threads), "--alt-ali", "10", "-a",
        ],
    )
    # 6. Filter
    _run_mmseqs(
        binary,
        [
            "filterresult", qdb, db_dir / f"{uniref_db}{db_suffix_seq}",
            work_dir / "res_exp_realign", work_dir / "res_exp_realign_filter",
            "--db-load-mode", db_load_mode, "--qid", "0", "--qsc", qsc,
            "--diff", "0", "--threads", str(threads), "--max-seq-id", "1.0", "--filter-min-enable", "100",
        ],
    )
    # 7. Result to MSA
    _run_mmseqs(
        binary,
        [
            "result2msa", qdb, db_dir / f"{uniref_db}{db_suffix_seq}",
            work_dir / "res_exp_realign_filter", work_dir / "uniref.a3m",
            "--msa-format-mode", "6", "--db-load-mode", db_load_mode, "--threads", str(threads),
        ]
        + filter_param,
    )
    # Cleanup intermediate results
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp_realign_filter"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp_realign"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res"])

    # 8. Env search (optional)
    if use_env:
        _run_mmseqs(
            binary,
            [
                "search", work_dir / "prof_res", db_dir / env_db,
                work_dir / "res_env", work_dir / "tmp3", "--threads", str(threads),
            ]
            + search_param,
        )
        _run_mmseqs(
            binary,
            [
                "expandaln", work_dir / "prof_res", db_dir / f"{env_db}{db_suffix_seq}",
                work_dir / "res_env", db_dir / f"{env_db}{db_suffix_aln}",
                work_dir / "res_env_exp",
                "-e", "inf", "--expansion-mode", "0",
                "--db-load-mode", db_load_mode, "--threads", str(threads),
            ],
        )
        _run_mmseqs(
            binary,
            [
                "align", work_dir / "tmp3/latest/profile_1", db_dir / f"{env_db}{db_suffix_seq}",
                work_dir / "res_env_exp", work_dir / "res_env_exp_realign",
                "--db-load-mode", db_load_mode, "-e", align_eval, "--max-accept", max_accept,
                "--threads", str(threads), "--alt-ali", "10", "-a",
            ],
        )
        _run_mmseqs(
            binary,
            [
                "filterresult", qdb, db_dir / f"{env_db}{db_suffix_seq}",
                work_dir / "res_env_exp_realign", work_dir / "res_env_exp_realign_filter",
                "--db-load-mode", db_load_mode, "--qid", "0", "--qsc", qsc,
                "--diff", "0", "--max-seq-id", "1.0", "--threads", str(threads), "--filter-min-enable", "100",
            ],
        )
        _run_mmseqs(
            binary,
            [
                "result2msa", qdb, db_dir / f"{env_db}{db_suffix_seq}",
                work_dir / "res_env_exp_realign_filter", work_dir / "bfd.mgnify30.metaeuk30.smag30.a3m",
                "--msa-format-mode", "6", "--db-load-mode", db_load_mode, "--threads", str(threads),
            ]
            + filter_param,
        )
        _run_mmseqs(binary, ["rmdb", work_dir / "res_env_exp_realign_filter"])
        _run_mmseqs(binary, ["rmdb", work_dir / "res_env_exp_realign"])
        _run_mmseqs(binary, ["rmdb", work_dir / "res_env_exp"])
        _run_mmseqs(binary, ["rmdb", work_dir / "res_env"])

    # 9. Template search (optional)
    if use_templates and (db_dir / f"{template_db}.dbtype").is_file():
        _run_mmseqs(
            binary,
            [
                "search", work_dir / "prof_res", db_dir / template_db,
                work_dir / "res_pdb", work_dir / "tmp2",
                "--db-load-mode", db_load_mode, "--threads", str(threads), "-a", "-e", "0.1",
            ]
            + template_search_param,
        )
        _run_mmseqs(
            binary,
            [
                "convertalis", work_dir / "prof_res", db_dir / f"{template_db}{db_suffix_tpl}",
                work_dir / "res_pdb", work_dir / template_db,
                "--format-output", "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar",
                "--db-output", "1", "--db-load-mode", db_load_mode, "--threads", str(threads),
            ],
        )
        _run_mmseqs(binary, ["rmdb", work_dir / "res_pdb"])
    elif use_templates:
        logger.info("Skipping template search: %s not found in %s", template_db, db_dir)

    # 10. Merge or move
    if use_env:
        _run_mmseqs(
            binary,
            ["mergedbs", qdb, work_dir / "final.a3m", work_dir / "uniref.a3m",
             work_dir / "bfd.mgnify30.metaeuk30.smag30.a3m"],
        )
        _run_mmseqs(binary, ["rmdb", work_dir / "bfd.mgnify30.metaeuk30.smag30.a3m"])
        _run_mmseqs(binary, ["rmdb", work_dir / "uniref.a3m"])
    else:
        _run_mmseqs(binary, ["mvdb", work_dir / "uniref.a3m", work_dir / "final.a3m"])

    # 11. Unpack
    _run_mmseqs(
        binary, ["unpackdb", work_dir / "final.a3m", work_dir, "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"]
    )
    _run_mmseqs(binary, ["rmdb", work_dir / "final.a3m"])

    if use_templates and (work_dir / template_db).with_suffix(".dbtype").is_file():
        _run_mmseqs(
            binary,
            ["unpackdb", work_dir / template_db, work_dir, "--unpack-name-mode", "0", "--unpack-suffix", ".m8"],
        )
        _run_mmseqs(binary, ["rmdb", work_dir / template_db])

    # Cleanup profiles and tmp dirs
    _run_mmseqs(binary, ["rmdb", work_dir / "prof_res"])
    _run_mmseqs(binary, ["rmdb", work_dir / "prof_res_h"])
    shutil.rmtree(work_dir / "tmp", ignore_errors=True)
    if use_templates:
        shutil.rmtree(work_dir / "tmp2", ignore_errors=True)
    if use_env:
        shutil.rmtree(work_dir / "tmp3", ignore_errors=True)


def _search_pair(
    binary: str,
    db_dir: Path,
    work_dir: Path,
    pairing_strategy: int,
    threads: int,
    gpu: bool,
    gpu_server: bool,
    pair_env: bool = False,
) -> None:
    """Run the paired MSA search pipeline.

    Args:
        pair_env: If True, use SPIRE environmental pairing database instead of UniRef30.
        gpu: Use GPU acceleration.
        gpu_server: Use dedicated GPU server.
    """
    uniref_db = "uniref30_2302_db"
    spire_db = "spire_ctg10_2401_db"
    qdb = work_dir / "qdb"

    # Select pairing database
    if pair_env:
        db = spire_db
        output_suffix = ".env.paired.a3m"
    else:
        db = uniref_db
        output_suffix = ".paired.a3m"

    db_load_mode = "2"
    db_suffix_seq = ".idx"
    db_suffix_aln = ".idx"
    if not (db_dir / f"{uniref_db}.idx").is_file() and not (db_dir / f"{uniref_db}.idx.index").is_file():
        db_load_mode = "0"
        db_suffix_seq = "_seq"
        db_suffix_aln = "_aln"

    max_seqs = "100000" if gpu else "10000"
    search_param: list[str | Path] = [
        "--num-iterations", "3", "--db-load-mode", db_load_mode, "-a", "-e", "0.1", "--max-seqs", max_seqs,
    ]
    if gpu:
        search_param += ["--gpu", "1", "--prefilter-mode", "1"]
    else:
        # Direct k-mer threshold to match ColabFold server (no quotes — see ColabFold #804)
        search_param += ["--prefilter-mode", "0", "--k-score", "seq:96,prof:80"]
    if gpu_server:
        search_param += ["--gpu-server", "1"]

    expand_param: list[str | Path] = [
        "--expansion-mode", "0", "-e", "inf", "--expand-filter-clusters", "0", "--max-seq-id", "0.95",
    ]

    _run_mmseqs(
        binary,
        ["search", qdb, db_dir / db, work_dir / "res", work_dir / "tmp", "--threads", str(threads)] + search_param,
    )
    _run_mmseqs(binary, ["mvdb", work_dir / "tmp/latest/profile_1", work_dir / "prof_res"])
    _run_mmseqs(binary, ["lndb", work_dir / "qdb_h", work_dir / "prof_res_h"])
    _run_mmseqs(
        binary,
        [
            "expandaln", qdb, db_dir / f"{db}{db_suffix_seq}",
            work_dir / "res", db_dir / f"{db}{db_suffix_aln}",
            work_dir / "res_exp",
            "--db-load-mode", db_load_mode, "--threads", str(threads),
        ]
        + expand_param,
    )
    _run_mmseqs(
        binary,
        [
            "align", work_dir / "prof_res", db_dir / f"{db}{db_suffix_seq}",
            work_dir / "res_exp", work_dir / "res_exp_realign",
            "--db-load-mode", db_load_mode, "-e", "0.001", "--max-accept", "1000000",
            "--threads", str(threads),
        ],
    )
    _run_mmseqs(
        binary,
        [
            "pairaln", qdb, db_dir / db,
            work_dir / "res_exp_realign", work_dir / "res_exp_realign_pair",
            "--db-load-mode", db_load_mode, "--pairing-mode", str(pairing_strategy),
            "--pairing-dummy-mode", "0", "--threads", str(threads),
        ],
    )
    _run_mmseqs(
        binary,
        [
            "align", work_dir / "prof_res", db_dir / f"{db}{db_suffix_seq}",
            work_dir / "res_exp_realign_pair", work_dir / "res_exp_realign_pair_bt",
            "--db-load-mode", db_load_mode, "-e", "inf", "-a", "--threads", str(threads),
        ],
    )
    _run_mmseqs(
        binary,
        [
            "pairaln", qdb, db_dir / db,
            work_dir / "res_exp_realign_pair_bt", work_dir / "res_final",
            "--db-load-mode", db_load_mode, "--pairing-mode", str(pairing_strategy),
            "--pairing-dummy-mode", "1", "--threads", str(threads),
        ],
    )
    _run_mmseqs(
        binary,
        [
            "result2msa", qdb, db_dir / f"{db}{db_suffix_seq}",
            work_dir / "res_final", work_dir / "pair.a3m",
            "--db-load-mode", db_load_mode, "--msa-format-mode", "5", "--threads", str(threads),
        ],
    )
    _run_mmseqs(
        binary,
        ["unpackdb", work_dir / "pair.a3m", work_dir, "--unpack-name-mode", "0", "--unpack-suffix", output_suffix],
    )
    # Cleanup
    _run_mmseqs(binary, ["rmdb", work_dir / "pair.a3m"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp_realign"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp_realign_pair"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_exp_realign_pair_bt"])
    _run_mmseqs(binary, ["rmdb", work_dir / "res_final"])
    _run_mmseqs(binary, ["rmdb", work_dir / "prof_res"])
    _run_mmseqs(binary, ["rmdb", work_dir / "prof_res_h"])
    shutil.rmtree(work_dir / "tmp", ignore_errors=True)


def _unpack_results(
    work_dir: Path, n_chains: int, use_pairing: bool, use_env_pairing: bool = False
) -> MsaResult:
    """Read unpacked A3M files into MsaResult."""
    unpaired: dict[int, str] = {}
    paired: dict[int, str] = {}
    m8_parts: list[str] = []
    for i in range(n_chains):
        a3m_path = work_dir / f"{i}.a3m"
        if a3m_path.is_file():
            unpaired[i] = a3m_path.read_text()
        else:
            unpaired[i] = ""
        if use_pairing:
            pair_path = work_dir / f"{i}.paired.a3m"
            # Merge env paired MSA into paired if present
            if use_env_pairing:
                env_pair_path = work_dir / f"{i}.env.paired.a3m"
                if env_pair_path.is_file() and pair_path.is_file():
                    with pair_path.open("a") as fp:
                        fp.write(env_pair_path.read_text())
                elif env_pair_path.is_file():
                    env_pair_path.rename(pair_path)
            if pair_path.is_file():
                paired[i] = pair_path.read_text()
            else:
                paired[i] = ""
        # Collect template m8 hits
        m8_path = work_dir / f"{i}.m8"
        if m8_path.is_file():
            m8_parts.append(m8_path.read_text())
    m8_text = "".join(m8_parts)
    return MsaResult(unpaired=unpaired, paired=paired, m8_text=m8_text)


def search_local(
    sequences: list[str],
    *,
    mmseqs: str = "mmseqs",
    db_dir: str | Path,
    use_env: bool = True,
    use_pairing: bool = False,
    use_env_pairing: bool = False,
    use_templates: bool = False,
    pairing_strategy: int = 0,
    filter: bool = True,
    threads: int = 8,
    gpu: bool = False,
    gpu_server: bool = False,
    tmpdir: str | Path | None = None,
    cache_dir: Path | None = None,
) -> MsaResult:
    """Search for MSAs using a local MMseqs2 binary.

    Args:
        sequences: List of amino acid sequences (one per chain).
        mmseqs: Path to the ``mmseqs`` binary.
        db_dir: Directory containing MMseqs2 databases (uniref30, envdb, pdb100).
        use_env: Search environmental database.
        use_pairing: Enable cross-chain pairing (multimer).
        use_env_pairing: Use SPIRE environmental pairing database.
        use_templates: Search PDB100 template database.
        pairing_strategy: Pairing mode (0=greedy, 1=complete).
        filter: Apply MSA filtering (qsc=0.8).
        threads: Number of threads.
        gpu: Use GPU acceleration (requires MMseqs2 with GPU support).
            Control which GPUs via ``CUDA_VISIBLE_DEVICES`` env var.
        gpu_server: Use dedicated GPU server for reduced per-request overhead
            (~1.5s savings). Requires a running ``mmseqs gpuserver`` process.
        tmpdir: Working directory (temporary if None).
        cache_dir: Optional directory for caching results.

    Returns:
        MsaResult with per-chain A3M text (and m8_text if use_templates).
    """
    db_dir = Path(db_dir)
    n_chains = len(sequences)

    # Check cache
    if cache_dir is not None:
        key = _cache_key(sequences, use_env, use_pairing)
        cached = _try_load_cache(cache_dir, key, n_chains)
        if cached is not None:
            return cached

    # Validate database existence
    if not (db_dir / "uniref30_2302_db.dbtype").is_file():
        raise FileNotFoundError(f"UniRef30 database not found at {db_dir}/uniref30_2302_db")
    if use_env and not (db_dir / "colabfold_envdb_202108_db.dbtype").is_file():
        raise FileNotFoundError(f"Environmental database not found at {db_dir}/colabfold_envdb_202108_db")

    cleanup = tmpdir is None
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="deepfold3_msa_")
    work_dir = Path(tmpdir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write query FASTA and create queryDB
        query_fasta = _build_query_fasta(sequences)
        query_file = work_dir / "query.fas"
        query_file.write_text(query_fasta)
        _run_mmseqs(mmseqs, ["createdb", query_file, work_dir / "qdb", "--shuffle", "0", "--dbtype", "1"])

        # Write lookup file
        with (work_dir / "qdb.lookup").open("w") as f:
            for i in range(n_chains):
                f.write(f"{i}\t{101 + i}\t0\n")

        # Run monomer search
        _search_monomer(mmseqs, db_dir, work_dir, use_env, use_templates, filter, threads, gpu, gpu_server)

        # Run pairing search (if multimer)
        if use_pairing and n_chains > 1:
            _search_pair(mmseqs, db_dir, work_dir, pairing_strategy, threads, gpu, gpu_server, pair_env=False)
            if use_env_pairing:
                _search_pair(mmseqs, db_dir, work_dir, pairing_strategy, threads, gpu, gpu_server, pair_env=True)

        result = _unpack_results(work_dir, n_chains, use_pairing and n_chains > 1, use_env_pairing)

        # Cleanup query DB
        _run_mmseqs(mmseqs, ["rmdb", work_dir / "qdb"])
        _run_mmseqs(mmseqs, ["rmdb", work_dir / "qdb_h"])

    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)

    # Save cache
    if cache_dir is not None:
        _save_cache(cache_dir, key, result)

    return result
