# Memory Tracing

When a model uses more device memory than you expect, the totals from `Device::measure_memory`
tell you how much is held, but not what holds it. On the CubeCL backends (CUDA, ROCm, WGPU, Metal,
CPU), memory tracing answers part of that question: the runtime's memory report then lists every
allocation that is still alive, with what it knows of the code that made it.

## Turning it on

Tracing is off by default. Turn it on in the `[cubecl]` section of `burn.toml` (or in
`cubecl.toml`):

```toml
[cubecl.memory]
tracing = "labels" # or "backtraces"
```

or for a single run, with an environment variable:

```sh
CUBECL_MEMORY_TRACING=backtraces cargo run --release
```

There are three levels:

- `disabled` (default): the report lists no allocations.
- `labels`: each allocation records the labels in effect when it was made (see below), the name of
  the thread that made it, and a sequence number giving its order among all allocations.
- `backtraces`: the same, plus the call stack that made the allocation. Capturing a stack costs a
  few microseconds per allocation, so use this level for debugging, not in normal runs. In release
  builds, enable line tables (`debug = "line-tables-only"` in your profile) to get file and line
  numbers.

## Labels

Loading a module from a record (`Module::load_record`) labels each parameter with its path, so the
weights of a loaded model are listed as `conv1.conv.weight`, `fc1.bias`, and so on.

You can label memory yourself. A label applies to every allocation made on the current thread from
the moment it is created until the variable holding it goes out of scope, and labels nest: an
allocation made inside `"model"` and then `"weights"` is listed under `["model", "weights"]`.

```rust, ignore
use burn::cubecl::allocation_label::AllocationLabel;

let _label = AllocationLabel::push("weights");
// allocations made on this thread are listed under "weights"
```

A label only reaches allocations made on the thread that created it. Most tensor operations do not
allocate on your thread: Burn hands them to the device's worker threads (named `DSU-…` and
`DSD-…`), and with kernel fusion it runs them later, when a result is read. Their allocations are
listed under the worker's thread name, without your labels, and their call stack ends inside Burn.
Loading a record allocates on your thread, which is why its labels reach the weights. So for now,
labels and call stacks name the weights of a loaded model and what your own thread allocates, and
the rest is grouped by worker thread.

## Reading the allocations

`Device::measure_memory` reports totals only. Read the allocations from the CubeCL client of the
device, here the first CUDA device:

```rust, ignore
use burn::cubecl::cuda::{CudaDevice, CudaRuntime};
use burn::cubecl::prelude::Runtime;
use burn::cubecl::MemoryScope;

let report = CudaRuntime::client(&CudaDevice::new(0)).memory_report(MemoryScope::Device);
for stream in &report.streams {
    let pools = &stream.pools;
    for pool in pools.dynamic.iter().chain([&pools.persistent, &pools.dedicated]) {
        for allocation in &pool.allocations {
            let origin = allocation.origin.as_ref();
            println!("{} bytes: {:?}", allocation.size, origin.map(|origin| &origin.labels));
        }
    }
}
```

A report only lists memory that is still in use: once the last tensor using an allocation is
dropped, the allocation is no longer listed.
