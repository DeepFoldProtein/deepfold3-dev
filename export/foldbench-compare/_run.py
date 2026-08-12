"""Shared run resolution + paths for the FoldBench comparison report.

The report is run-agnostic: external-model rows are fixed reference data under `data/`,
and every figure is generated for one DeepFold3 bench series into `out/<short>/`.
Switch runs with `--run`; nothing in this directory hardcodes a series.

    from _run import add_args, resolve
    ap = argparse.ArgumentParser(); add_args(ap)          # --run / --step / --after
    A = ap.parse_args(); R = resolve(A)
    R.series      # 'df3v5r15'  — bench/runs/<series>
    R.short       # 'r15'       — out/<short>/, filename suffixes, CSV column names
    R.label       # 'DF3r15'    — figure legend / axis label
    R.eval_dir    # bench/runs/<series>/eval
    R.out_dir     # export/foldbench-compare/out/r15   (created on resolve)
    R.latest_step(cat_full)
"""
import csv
import glob
import os
import re
from collections import defaultdict

REPO = "/home/jovyan/work/deepfold3-dev"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = f"{REPO}/bench/runs"
TGT = f"{REPO}/bench/data/targets"
DATA = f"{HERE}/data"                                   # fixed external reference
EXT_CACHE = f"{DATA}/foldbench_ext_cache.json"          # tarball parse (mean/oracle/rank-1)
EXT_SUMMARY = f"{DATA}/foldbench_all_models_per_target.csv"   # FoldBench's own per-target summary
DATES = f"{DATA}/foldbench_target_dates.csv"            # RCSB release dates, for --after

DEFAULT_RUN = "df3v5r15"


class Run:
    """One DeepFold3 bench series, with the paths and display names derived from it."""

    def __init__(self, series):
        self.series = series
        self.eval_dir = f"{RUNS}/{series}/eval"
        m = re.search(r"(r\d+)$", series)                # df3v5r15 -> r15; df3v2 -> df3v2
        self.short = m.group(1) if m else series
        self.label = f"DF3{self.short}" if m else series.upper()
        self.out_dir = f"{HERE}/out/{self.short}"
        os.makedirs(self.out_dir, exist_ok=True)

    def latest_step(self, cat_full, marker="summary.csv"):
        """Newest benched step for a category. `summary.csv` is written last, so its
        presence is the completion marker; pass marker='raw' when only raw CSVs matter."""
        pat = (f"{self.eval_dir}/{cat_full}/step_*/raw/{cat_full}_ost.csv"
               if marker == "raw" else f"{self.eval_dir}/{cat_full}/step_*/{marker}")
        steps = [int(re.search(r"step_(\d+)", p).group(1)) for p in glob.glob(pat)]
        return max(steps) if steps else None

    def raw_csv(self, cat_full, step):
        return f"{self.eval_dir}/{cat_full}/step_{step}/raw/{cat_full}_ost.csv"

    def summary_csv(self, cat_full, step):
        return f"{self.eval_dir}/{cat_full}/step_{step}/summary.csv"

    def out(self, name):
        return f"{self.out_dir}/{name}"

    def __repr__(self):
        return f"Run({self.series} -> {self.label}, out/{self.short})"


def add_args(ap, after=True):
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help=f"bench series under bench/runs/ (default: {DEFAULT_RUN}). "
                         "Accepts the short form too, e.g. 'r15'.")
    ap.add_argument("--step", type=int, default=None,
                    help="pin a checkpoint step (default: latest benched, per category)")
    if after:
        ap.add_argument("--after", default=None, metavar="YYYY-MM-DD",
                        help="keep only targets released AFTER this date, e.g. 2023-06-01 "
                             "(Boltz-2 cutoff)")
    return ap


def resolve(A):
    """--run value -> Run. Accepts a series dir name or its short 'rNN' form."""
    want = A.run
    if os.path.isdir(f"{RUNS}/{want}"):
        return Run(want)
    hits = sorted(d for d in os.listdir(RUNS)
                  if os.path.isdir(f"{RUNS}/{d}") and re.search(r"(r\d+)$", d)
                  and re.search(r"(r\d+)$", d).group(1) == want)
    if len(hits) == 1:
        return Run(hits[0])
    avail = ", ".join(sorted(d for d in os.listdir(RUNS) if os.path.isdir(f"{RUNS}/{d}/eval")))
    raise SystemExit(f"--run {want!r}: {'ambiguous, matches ' + str(hits) if hits else 'no such series'}. "
                     f"Available: {avail}")


def load_dates(after):
    """pdb_id -> release date, only when --after is in play (otherwise an empty dict)."""
    d = {}
    if after and os.path.isfile(DATES):
        for r in csv.DictReader(open(DATES)):
            if r.get("released"):
                d[r["pdb_id"]] = r["released"]
    return d


def designated(cat_full):
    """target -> the single FoldBench-designated interface chain pair."""
    return {r["pdb_id"]: frozenset([r["interface_chain_id_1"], r["interface_chain_id_2"]])
            for r in csv.DictReader(open(f"{TGT}/{cat_full}.csv"))}


def annotated(cat_full):
    """target -> set of ALL annotated interface chain pairs (targets can have several)."""
    d = defaultdict(set)
    for r in csv.DictReader(open(f"{TGT}/{cat_full}.csv")):
        d[r["pdb_id"]].add(frozenset([r["interface_chain_id_1"], r["interface_chain_id_2"]]))
    return d


def rank1_value(scored):
    """Expected DockQ of the confidence-selected ("rank-1") sample.

    `scored` = [(ranking_score, quality), ...] for one target. Returns the mean quality
    over every sample TIED at the top score, or None if nothing is scored.

    Why a mean and not a pick: `ranking_score` is stored to 4 decimals, and a converged
    DeepFold3 run's scores for one target sit inside ~5e-4 of each other — so at the
    available precision 109/211 protein-protein targets at r14 step 49k have a tied top
    score, half of them with 5+ samples tied. The tie-break rule is therefore not a detail:

      * picking the tied sample with the best DockQ (what plot_foldbench_ranked.py used to
        do, via a tuple `max`) is a mini-oracle — it peeks at the answer, and inflated r14's
        protein-protein High tier by 3.4 pts over an unbiased rule;
      * picking the first tied row (what plot_foldbench_conf.py used to do) is unbiased but
        depends on CSV row order, so the two figures disagreed on the same quantity;
      * averaging over the tied set is what you actually get in expectation when you deploy
        a ranking that cannot distinguish those samples, and it is order-independent.

    External models are unaffected in practice (their scores are near-unique), but they go
    through this same function so the comparison is rule-for-rule identical.
    """
    if not scored:
        return None
    top = max(s for s, _ in scored)
    tied = [q for s, q in scored if s == top]
    return sum(tied) / len(tied)


def tiers(vals):
    """CAPRI tier percentages (acceptable, medium, high, success=sum) over DockQ values."""
    n = len(vals)
    if not n:
        return 0.0, 0.0, 0.0, 0.0
    a = sum(1 for d in vals if 0.23 <= d < 0.49) / n * 100
    m = sum(1 for d in vals if 0.49 <= d < 0.80) / n * 100
    h = sum(1 for d in vals if d >= 0.80) / n * 100
    return a, m, h, a + m + h


CATS_PPI_AGAB = [("ppi", "interface_protein_protein", "protein–protein"),
                 ("agab", "interface_antibody_antigen", "antibody–antigen")]

# Shared figure palette. Ours is orange, external models blue.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
TIER = ["#bcd4ec", "#5a9bd4", "#1f5c9e"]
OURTIER = ["#f6c3a6", "#f0946a", "#c1490f"]
OURCOL = "#eb6834"
