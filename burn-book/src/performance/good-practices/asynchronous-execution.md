# Asynchronous Execution

Most Burn backends execute tensor operations in an asynchronous manner. However, the async notation
is often not required for most tensor operations, privileging the simplicity of sync Rust.

There are only a few operations that trigger synchronization of the backend, and it is very
important to correctly handle those to optimize hardware utilization. Those operations are
`into_data`, `into_scalar`, and `sync`. Some tensor operations might call `into_data` underneath,
triggering a synchronization, like `to_device` for some backends.

There are several ways to minimize synchronization overhead, one of which is to batch sync
operations into a single transaction. Burn provides a high-level composable API to build
transactions, which will only trigger a single sync on the device.

For instance, it is often used when collecting metrics during training:

```rust
/// All of these variables are tensors.
let (output, loss, targets) = ..;

/// Now output, loss, and targets will be `TensorData` stored on the CPU.
let [output, loss, targets] = Transaction::default()
    .register(output)
    .register(loss)
    .register(targets)
    .execute()
    .try_into()
    .expect("Correct amount of tensor data");
```

Another way of optimizing reads and avoiding device stalls is to read the data on a different
thread. Under the hood, CubeCL-based backends assign different execution queues for different
threads, meaning that syncing a thread shouldn’t impact the throughput of another thread.

## Using Different Backends for Different Tasks

Tensor operations aren’t the only things that are asynchronous; dataset and dataloading are also
lazily executed. This allows for efficient data augmentation and sampling without having to cache
huge datasets on disk. However, this might reduce training throughput if data augmentation is
performed on the same device as the training itself. So, it is normally encouraged to use a
different device, maybe even a different backend, for that purpose. For optimal performance, also
avoid small allocations followed by a batching procedure. Even if it doesn’t break asynchronicity,
it can slow down performance.

```rust
/// Items is a vector of many tensors.
let items = ..;
let batch = Tensor::cat(items, 1);
```

Prefer doing the concatenation of tensors on the data augmentation device and not on the training
device.

```rust
/// Items is a vector of many tensors.
let items = ..;
let device_training = ..;
let axis_batch = 0;

let items = Tensor::cat(items, axis_batch);
let batch = Tensor::from_data(items.into_data(), device_training);
```
