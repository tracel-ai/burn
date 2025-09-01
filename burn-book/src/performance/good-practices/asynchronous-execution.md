# Asynchronous Execution

Most burn backends execute tensor operations in an asynchronous manner. However, the async notation
is often not required for most tensor operations, privilaging the simplicity of sync Rust.

There are only a few operations that triggers synchronozation of the backend, and it is very
important to correctly handle those to optimize optimal hardware utilization. Those operations are
`into_data`, `into_scalar` and `sync`. Some tensor operation might call `into_data` underneats,
triggering a synchronozation, like `to_device` for some backend.

There are many different way to optimize those syncs, and the first one is actually to group them
into a single transaction. Burn provide a high level composable API to build transaction, which will
only trigger a single sync on the device.

For instance it is often used when collecting metrics during training:

```rust
/// All of those variables are tensors.
let (output, loss, targets) = ..;

/// Now output, loss and targets will be `TensorData` stored on the CPU.
let [output, loss, targets] = Transaction::default()
    .register(output)
    .register(loss)
    .register(targets)
    .execute()
    .try_into()
    .expect("Correct amount of tensor data");
```

Another way of optimizing reads and avoid device stall is to read the data on a different thread.
Under the hood, CubeCL based backends assign different execution queue for different thread, meaning
that synching a thread shouldn't impact the throughput of another thread.

## Using different backend for different tasks

Tensor operation isn't the only thing that is asynchronous, dataset and dataloading is also lazily
executed. This allow for efficient data augmentation and sampling without having to cache huge
dataset on disk. However, this might reduce training throughput if data augmentation is performed on
the same device as the training itself. So it is normally encourage to use a different device, maybe
even a different backend for that purpose. For optimal performance, also avoid small allocations
followed by a batching procedure. Even if it doesn't break asynchronousity, it can slowdown
performance.

```rust
/// Items is a vector of many tensors.
let item = ..;
let batch = Tensor::cat(items, 1);
```

Perfer doing the concatenation of tensor on the data augmentation device and not on the training
device.

```rust
/// Items is a vector of many tensors.
let item = ..;
let device_training = ..;
let device_data = ..;
let axis_batch = 0;


let items = Tensor::cat(items, axis_batch);
let batch = Tensor::from_data(items.into_data(), device_training);
```
