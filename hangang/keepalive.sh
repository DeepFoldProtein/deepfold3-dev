#!/usr/bin/env bash
# Report (and optionally refresh) access times under $DF3_ROOT.
#
# Hangang purges /scratch files whose atime is older than 15 days: they are
# renamed to ToBeDelete_* and deleted some 20-30 days later. Training reads
# keep the shards it touches fresh, but a pause between runs, an unused pack
# (e.g. eval-only data) or the venv can age out — and losing one shard of a
# pack makes every record in it unreadable.
#
#   bash hangang/keepalive.sh            # report: files older than 10 days, ToBeDelete_* renames
#   bash hangang/keepalive.sh --touch    # also refresh atime on every file (a few thousand: cheap)
#
# Check KISTI's policy before automating --touch (consult@ksc.re.kr): sites
# often forbid defeating the purge. The safe fallback is re-pushing what
# aged out: hangang/transfer.sh status + push resend only missing files.

set -euo pipefail
source "$(dirname "$0")/env.sh"

old=$(find "${DF3_ROOT}" -type f -atime +10 | wc -l)
doomed=$(find "${DF3_ROOT}" -name 'ToBeDelete_*' | wc -l)
echo "under ${DF3_ROOT}: ${old} file(s) not read for >10 days, ${doomed} already renamed ToBeDelete_*"
if ((doomed)); then
    echo "Restore with: find ${DF3_ROOT} -name 'ToBeDelete_*' -exec sh -c 'mv \"\$1\" \"\$(dirname \"\$1\")/\$(basename \"\$1\" | sed s/^ToBeDelete_//)\"' _ {} \\;"
fi
if [[ "${1:-}" == "--touch" ]]; then
    find "${DF3_ROOT}" -type f ! -name 'ToBeDelete_*' -exec touch -a -c {} +
    echo "atime refreshed"
fi
