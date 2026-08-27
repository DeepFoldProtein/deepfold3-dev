#!/usr/bin/env bash
# Map a job's pods to their RDMA (multus) IPs, and optionally resolve the
# address an NCCL bootstrap failure named to the host that owns it.
#
# Why this exists: an NCCL startup failure reports the peer it could not reach
# as an RDMA IP (10.1.x.x), not a pod IP (192.168.x.x) and not a hostname. Those
# addresses are assigned per pod, so they change every submission and cannot be
# resolved after the job tears down — which is exactly when you want them. Run
# this WHILE the pods are still up.
#
# Usage:
#   bash kakao/rdma_ip_map.sh df3-stage0-r19              # dump the whole map
#   bash kakao/rdma_ip_map.sh df3-stage0-r19 10.1.64.7    # resolve one address
#
# Typical flow when a run dies in ncclOsSocketPollConnect:
#   1. resubmit
#   2. run this immediately, redirecting to a file, so the map survives teardown
#   3. grep the training log for the address NCCL named
#   4. feed that address back in as the second argument
#
# If the address resolves to a host repeatedly across submissions, that host is
# a genuine candidate for the nodeAffinity NotIn list in
# trainjob-multinode.yaml (BOTH the launcher and node blocks). If it resolves to
# a DIFFERENT host each time, the problem is a startup race rather than a bad
# node, and excluding hosts will not help.

set -euo pipefail

JOB="${1:?usage: rdma_ip_map.sh <job-name-prefix> [ip-to-resolve]}"
TARGET="${2:-}"
NAMESPACE="${NAMESPACE:-kbm-g-np-kias}"
KUBECTL="${KUBECTL:-kubectl -n ${NAMESPACE}}"

${KUBECTL} get pods -o json | python3 -c "
import json, re, sys

job, target = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '')
data = json.load(sys.stdin)

owner = {}
rows = []
for pod in data['items']:
    name = pod['metadata']['name']
    if job not in name:
        continue
    node = pod['spec'].get('nodeName') or '?'
    phase = pod['status'].get('phase') or '?'
    ann = pod['metadata'].get('annotations', {}).get('k8s.v1.cni.cncf.io/network-status', '')
    ips = sorted(set(re.findall(r'10\.\d+\.\d+\.\d+', ann)))
    for ip in ips:
        owner[ip] = (name, node)
    rows.append((name, node, phase, ips))

if not rows:
    print(f'no pods matching {job!r} (already torn down?)')
    raise SystemExit(1)

for name, node, phase, ips in sorted(rows):
    span = f'{ips[0]} .. {ips[-1]}' if ips else '(no rdma ips)'
    print(f'{name:44s} {node:18s} {phase:10s} {len(ips):2d}  {span}')

if target:
    hit = owner.get(target)
    print()
    print(f'{target} -> {hit[1]} ({hit[0]})' if hit else
          f'{target} -> NOT owned by any {job} pod '
          f'(stale address, or a pod that never registered its RDMA interface)')
" "$JOB" "$TARGET"
