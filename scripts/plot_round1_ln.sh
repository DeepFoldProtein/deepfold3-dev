#!/usr/bin/env bash
# Plot Round 1 regularizer ablation.
#
# Emits into runs/round1_ln/:
#   1s_overlay.png   — all present arms, 1-sample
#   5s_overlay.png   — all present arms, 5-sample
#   combined.png     — all present 1s + 5s runs (solid=1s, dashed=5s)
#
# Arms are included automatically based on which ``{wave}_{arm}`` directories
# exist. Supported arms: noreg, pair, ln, ln_pair, single, single_pair.
#
# Usage:
#   scripts/plot_round1_ln.sh

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PLOT="python scripts/plot_overfit.py"
OUT="runs/round1_ln"

# (arm_name, color) pairs — consistent across both waves.
ARMS_META=(
    "noreg:C0"
    "pair:C1"
    "ln:C2"
    "ln_pair:C3"
    "single:C4"
    "single_pair:C5"
)

# Prefer the ln_pair label shortening.
label_for() {
    case "$1" in
        ln_pair)     echo "ln+pair" ;;
        single_pair) echo "single+pair" ;;
        *)           echo "$1" ;;
    esac
}

build_wave_args() {
    # Args: wave linestyle prefix_label
    local wave="$1"
    local linestyle="$2"
    local with_wave_prefix="$3"
    local -a args=()
    for entry in "${ARMS_META[@]}"; do
        local arm="${entry%%:*}"
        local color="${entry##*:}"
        local d="$OUT/${wave}_${arm}"
        if [[ -d "$d" ]]; then
            local lbl
            lbl="$(label_for "$arm")"
            if [[ "$with_wave_prefix" == "1" ]]; then
                lbl="${wave}_${lbl}"
            fi
            args+=("${d}=${lbl}=${color}${linestyle}")
        fi
    done
    printf '%s\n' "${args[@]}"
}

# Collect present args for each wave.
mapfile -t ARGS_1S < <(build_wave_args 1s "-" 0)
mapfile -t ARGS_5S < <(build_wave_args 5s "-" 0)
mapfile -t ARGS_1S_PREFIXED < <(build_wave_args 1s "-" 1)
mapfile -t ARGS_5S_PREFIXED < <(build_wave_args 5s "--" 1)

if (( ${#ARGS_1S[@]} > 0 )); then
    $PLOT \
        --out "$OUT/1s_overlay.png" \
        --title "Round 1 — 1-sample (lr=1e-4, seed=99)" \
        "${ARGS_1S[@]}"
fi

if (( ${#ARGS_5S[@]} > 0 )); then
    $PLOT \
        --out "$OUT/5s_overlay.png" \
        --title "Round 1 — 5-sample (lr=1e-4, seed=99)" \
        "${ARGS_5S[@]}"
fi

if (( ${#ARGS_1S_PREFIXED[@]} + ${#ARGS_5S_PREFIXED[@]} > 0 )); then
    $PLOT \
        --out "$OUT/combined.png" \
        --title "Round 1 — regularizer ablation (solid=1s, dashed=5s)" \
        "${ARGS_1S_PREFIXED[@]}" "${ARGS_5S_PREFIXED[@]}"
fi

echo
echo "Plots written to $OUT/{1s_overlay,5s_overlay,combined}.png"
