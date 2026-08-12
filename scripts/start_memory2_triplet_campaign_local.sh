#!/bin/bash
# Start the resumable local campaign exactly once and return to the shell.
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_root"

root=output/memory2_triplet_campaign_20260809
log="$root/overnight_anchor.log"
pid_file="$root/overnight_anchor.pid"
mkdir -p "$root"

if [[ -s "$pid_file" ]]; then
    old_pid=$(cat "$pid_file")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "campaign already running as PID $old_pid"
        exit 1
    fi
fi

nohup env \
    PMM_JOBS="${PMM_JOBS:-12}" \
    PMM_MEMBERS_PER_BATCH="${PMM_MEMBERS_PER_BATCH:-4}" \
    caffeinate -i bash scripts/run_memory2_triplet_campaign_local.sh \
    > "$log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$pid_file"
echo "Campaign PID: $pid"
echo "Log: $log"
