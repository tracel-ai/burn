# Burn Shard

> Splits [Burn](https://github.com/tracel-ai/burn) tensors over several devices, for tensor parallelism

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-shard.svg)](https://crates.io/crates/burn-shard)
[![Documentation](https://docs.rs/burn-shard/badge.svg)](https://docs.rs/burn-shard)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

**Experimental:** the API may change, and so far only the forward pass of a Megatron-style MLP
keeps its tensors split.

A sharded compute device is one device standing for several. Each tensor on it has a `Sharding`: replicated
on every member, split along one dim, or a partial sum of the members' shards. An op with a
sharding rule runs on every member's shard at once. Any other op gathers its inputs and runs whole
on every member, so results stay correct while rules are added.

`ShardBackend` runs a sharded compute device as a [burn-router](https://github.com/tracel-ai/burn/tree/main/crates/burn-router)
backend: one `TensorInterpreter` per member replays each op at the member's local shapes, and
shards move between members with the backend's collectives.

## Usage

Enable Burn's `shard` feature, with the backend its members run:

```toml
burn = { version = "0.23", features = ["shard", "cuda"] }
```

```rust,ignore
use burn::prelude::*;
use burn::tensor::{Distribution, Sharding::Split, Shards, activation};

let device = Device::shard_options(&[Device::cuda(0), Device::cuda(1)]).init()?;
let members = device.shard_members().unwrap();

// Each member makes its own shard, so no device ever holds a whole weight.
let split = |shape: [usize; 2], dim| {
    let shards = members
        .iter()
        .map(|member| Tensor::<2>::random(shape, Distribution::Default, member))
        .collect();
    Tensor::from_shards(Shards { shards, sharding: Split { dim } }, &device)
};
// Megatron's MLP: the first weight split by column, the second by row.
let w1 = split([1024, 2048], 1)?;
let w2 = split([2048, 1024], 0)?;

let x = Tensor::<2>::random([8, 1024], Distribution::Default, &device);
// Split along its last dim, with nothing sent between members.
let hidden = activation::relu(x.matmul(w1));
// A partial sum of the members' shards, reduced when an op reads it whole.
let output = hidden.matmul(w2);
```

A sharded compute device's members are Cube devices, of one runtime or of several, or Flex
devices. Members on different runtimes form a sharded compute device whose every collective is
emulated through the host: correct, and far slower than members that share a runtime.

The caller makes a split tensor's shards, each on its member, and `Tensor::from_shards` checks
that they make up one tensor: shard `i` on member `i`, one dtype, and the balanced chunk of the
split dim for each member, the first members one longer when it does not divide. `DimSplit` gives
each member's range of that dim and cuts a `TensorData` to it. A tensor made from data or by an op
on the sharded compute device itself is replicated, and a random one is drawn on the first member
and copied to the others.

`Sharding::ReplicatedSplit { dim, groups }` cuts the dim into `groups` chunks instead, each held
by a contiguous group of members: two key and value heads over four members are
`ReplicatedSplit { dim: 1, groups: 2 }`, members 0 and 1 holding the first. Which cards pair up
is set by the order the members are given to `shard_options`.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
