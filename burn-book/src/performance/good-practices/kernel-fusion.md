# Kernel Fusion

An interesting properties of async execution is that it allows performance optimization like kernel
fusion. Couple with CubeCL and its Just-In-Time compiler, Burn is able to serializse tensor
operations into a symbolic graph, then optimizing it for improved efficiency.

Kernel fusion might reorder operations to reduce global memory reads, writes and allocations. Being
aware of what operations can be fused is relevant, as it can be easy to break a graph of execution.

The easiest way of optimizing for fusion is to avoid keeping tensor alive for too long, as when
fusion isn't possible, all tensors that will be used afterward will triggers a global memory write.
Fortunatly Rust and Clippy are quite good at detecting unecessary clones, but a special care could
still be taken.

View operations can also interphere with fusion. They can be included in optimized graphs, but in a
limited amount and they reduce vectorization potential as we have less guarantees about memroy
access patterns with transformed indices. So it is a good practice to group those view operations
together before execute block of operations.

```rust
let tensor4 = tensor1.unsqueeze().matmul(tensor2) + tensor3.unsqueeze();

```

Could be improved with the following:

```rust

let tensor1 = tensor1.unsqueeze();
let tensor3 = tensor3.unsqueeze();
let tensor4 = tensor1.matmul(tensor2) + tensor3;
```

This reduces the necessary reordoring and might reduce a global memory write or improve
vectorization. We might be able to detech those patterns in the future, but for now it's a good
ideas to order your operations using this pattern. For a reminder, view operations normally may only
update tensor metadata in most cases, those operations are `slice`, `slice_assign`, `select`,
`gather`, `scatter`, `reshape`, `swap_dims`, `transpose`, `unsqueeze`, etc.

With fusion enabled, is it often not necessary to write custom kernels, as you can rely on our
system to optimize most elemenwise operations. However, most compute-bound kernels requires many
tricks and deep knowledge of GPU memory architectures, where automatic compiler optimizations often
underperform compared to human algorithms design. This is why Burn's approach to fusion is centered
around fuse-on-read and fuse-on-write. Meaning that complex compute-bound kernels that change the
shapes of tensors can fuse a block of elemenwise operations when reading the input tensor and when
writing the output tensor. The implications is that multiple compute bounds operations in a sqeuence
can reduce fusion potential.

```rust
// That line might trigger 3 writes: tensor1, tensor2 and tensor3, if tensor1 and tensor2 are abstract tensor.
//
// If they are actually concrete in global memory, th
let tensor3 = tensor1.clone().sum_dim(tensor2.clone(), 2);

let tensor4 = tensor2.sum_dim(tensor3, 2);
let tensor5 = tensor4 + (tensor1 * tensor2);
```

```rust
let tmp = tensor1.clone() + tensor2.clone();
let tensor3 = tensor1.sum_dim(tensor2, 2);
let tensor4 = tensor2.sum_dim(tensor3, 2);
let tensor5 = tensor4 + tmp;
```

The lesson? Try when possible to give the only lasted value to a compute operation, don't clone a tensor before a compute bound operations, as it might trigger an additional write if that tensor isn't actually being materialized from inital fusion.

It's a bit complex, but the first code snippet is actually better if `tensor1` and `tensor2` are concrete in global memory.
It would be the case if `tensor1` and `tensor2` are model paramters, there perfer this implementation style when faced with similar scenario.

The second code snippet is preferred when `tensor1` and `tensor2` are virtual tensors, meaning that they were fused by earlier operations and requires a globsl memory read to be access later one.
This happens if those tensors are part of a signal in neural networks.

Reordering of operation can help in such scenario, but will not create temporary values, therefore making the previous optimization harder.
We might at some point automatically optimize those cases, but the solution space is quite huge, and it's not a planned optimization.
Profiling model blocks is always a good ideas to identigy which code block is faster when faced with ambiguous situations.
