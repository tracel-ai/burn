# Remote Training

Trains the MNIST model from the [`mnist`](../mnist) example data-parallel across the devices
of one or more `burn-remote` servers, over WebSocket or Iroh. It exists to exercise and
measure the multi-device remote topologies described in the
[distributed computing chapter](../../burn-book/src/performance/distributed-computing.md).

## Roles

One binary, two roles. The server hosts every device its compiled backend enumerates:

```sh
cargo run --release --example remote-training --features vulkan -- server ws://0.0.0.0:3000
cargo run --release --example remote-training --features cuda -- server iroh://my-rig
```

The client trains on whatever `--server` specs it is given. A spec is
`ws://host:port[#i,j]` or `iroh://topic#i,j`, where the fragment selects device indices on
that server (omitted for WebSocket, it means every hosted device; Iroh requires it):

```sh
cargo run --release --example remote-training -- train \
    --server ws://192.168.2.139:3000 --strategy multi
```

An Iroh topic stands in for the server identity on both sides: anyone who knows the topic
can host under it, which suits an experiment on a trusted network and nothing else.

## Strategies

| `--strategy` | What happens | Constraints |
|---|---|---|
| `single` | Everything on the first device. | |
| `multi` | Data-parallel; gradients gathered and optimized on the first device. | Re-uploads the model to every device each step. |
| `multi-sharded` | Data-parallel; optimizer state sharded across devices. | Same re-upload. |
| `ddp` | Data-parallel; gradients all-reduced on the server. | One server, and a backend with collectives (CUDA today). |

Cross-server gradient movement in `multi` goes server-to-server; the client never sees the
payload. Cross-server `ddp` is rejected up front because the remote client does not support
cross-peer all-reduce.

## Topologies worth comparing

On a machine with 2 NVIDIA + 2 AMD GPUs, all visible through Vulkan:

```sh
# One Vulkan server hosting all four GPUs.
... --features vulkan -- server ws://0.0.0.0:3000
... -- train --server ws://rig:3000 --strategy multi

# Four processes, one GPU each: every process hosts all four devices, so the client
# takes device i from server i.
... --features vulkan -- server ws://0.0.0.0:300$i   # i = 0..3
... -- train --server 'ws://rig:3000#0' --server 'ws://rig:3001#1' \
             --server 'ws://rig:3002#2' --server 'ws://rig:3003#3' --strategy multi

# CUDA server on the NVIDIA pair, with server-side all-reduce.
... --features cuda -- server ws://0.0.0.0:3000
... -- train --server ws://rig:3000 --strategy ddp
```

Scaling runs cap the device count by listing indices: `ws://rig:3000#0`, `#0,1`, `#0,1,2,3`.

The run prints wall-clock throughput at the end; `RUST_LOG=info,burn_remote=debug` shows
per-connection activity.

## Current status

Verified on a 4-GPU machine (2x NVIDIA, 2x AMD, all via Vulkan; NVIDIA pair via CUDA):

- `single` works over both transports. Iroh measured ~30% faster than WebSocket on the
  same device and workload.
- `multi` across one server process per GPU works from a high-latency client, but a
  low-latency (loopback) client exposes an ordering race in the server-to-server tensor
  transfer: runs crash with uninitialized-memory errors or deadlock on the first step.
- `multi` through a single multi-GPU server requires in-process cross-device transfers,
  which the wgpu runtime does not implement (CUDA only); on CUDA it hits a collective
  communicator initialization race between sessions.
- `ddp` hits the same communicator race and does not currently complete against a CUDA
  server.

Until those stack issues are fixed, treat the multi-device strategies as reproducers for
them rather than working configurations.
