# 12-node trainjob stuck — Kakao Cloud GPU capacity insufficient

**Date**: 2026-04-26
**Trainjob**: `df3-20260426-012311`
**Outcome**: stuck in init, never reached step 1; needs cancel.

## Symptoms

5 minutes after `N_NODES=12 bash kakao/submit.sh`:

- 8 worker pods `Running` (node-0-{0,1,2,3,4,5,6,8})
- 2 worker pods `Pending` (node-0-7, node-0-10)
- 1 worker pod `ImagePullBackOff` (node-0-9)
- 1 launcher pod `CrashLoopBackOff` (5 restarts and counting)

## Root cause

**The cluster does not have 12 nodes of free GPU capacity.** Pending pods report:

```
0/272 nodes are available:
  124 Insufficient nvidia.com/gpu
   25 Insufficient nvidia.com/mlnxnics
   31 node(s) were unschedulable
   62 Insufficient memory
  117 node(s) didn't match Pod's node affinity/selector
    1 Insufficient cpu
preemption: 0/272 nodes are available:
  124 No preemption victims found for incoming pod
```

`cluster-autoscaler` declined: `pod didn't trigger scale-up` — the scaler determined no scale-up could satisfy the request, so the pending pods will not get nodes automatically.

The launcher's repeated restarts are downstream — `mpirun -np 12` fails because DNS for the never-spawned pods (node-0-7, 10) doesn't resolve. Each launcher restart re-attempts and re-fails after the wait-for-rendezvous timeout.

The `node-0-9 ImagePullBackOff` is unrelated — a transient registry hiccup; would have resolved on retry.

## Capacity at time of submit

Effective fleet: ~**10 schedulable nodes** with 8× B200 + 8× mlnxnics each. The cluster has 272 total nodes, but the GPU/IB-NIC subset matching our pod spec (8 nvidia.com/gpu + 8 nvidia.com/mlnxnics + memory + the existing PVC mounts) was at 8/10 utilization at the time of submit, with no spare capacity for our 12-node ask.

## What this teaches us about N_NODES selection

The kakao trainer guide doesn't expose a "free GPU node count" query through the SA we use, so submitting blind at N_NODES=12 was speculative. **Pre-flight check** that would have caught this:

```bash
# Doesn't work for us — SA cannot list nodes:
kubectl get nodes -l 'nvidia.com/gpu.product=B200' -o wide

# What does work — count concurrent trainjob pods consuming GPU:
kubectl -n kbm-g-np-kias get pods -o json \
  | jq '[.items[] | select(.status.phase == "Running")
                 | select(.spec.containers[].resources.requests."nvidia.com/gpu" // false)
                 | .metadata.name] | length'
```

If we had checked, we'd have seen 8 nodes already in use by other workloads.

**Empirical heuristic**: 8 nodes (= 64 ranks) is the historically-known-safe size on this cluster. 12 nodes occasionally works during low-utilization periods; 16 nodes (the original numNodes in the YAML before this session) was used in 051858. Above 8 → check capacity first.

## What to do when this happens

Two clean recoveries:

1. **Drop to known-good size**: cancel and resubmit at `N_NODES=8`. Trade-off: smaller effective batch (`8×8×BS×accum`).

2. **Match batch via accumulate**: cancel and resubmit at `N_NODES=8` with `accumulate_grad_batches=3` to recover `8×8×2×3 = 384` effective batch. This preserves the target batch we wanted from 12 nodes; cost is +50% wall-clock per optimizer step.

Avoid: waiting for capacity to free. The autoscaler said it can't help, and other workloads holding capacity are typically also long-running training.

## Action taken

(To be filled in once user picks recovery path.)

## See also

- `kakao/trainjob-multinode.yaml` — the N_NODES placeholder lives in 3 spots (numNodes, mpirun -np, torchrun --nnodes), all substituted by `submit.sh`
- `kakao/submit.sh` — accepts `N_NODES=N` env var override
- `reference_kakao_broken_host.md` (memory) — host-10-0-1-5 has a known-bad GPU; we removed the affinity exclusion this session, so pods can now land there
