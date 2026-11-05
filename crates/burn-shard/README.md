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
use burn::tensor::{Distribution, Sharding::Split, activation};

let device = Device::shard_options(&[Device::cuda(0), Device::cuda(1)]).init()?;

// Megatron's MLP: the first weight split by column, the second by row.
let w1 = Tensor::<2>::random([1024, 4096], Distribution::Default, &device).with_sharding(Split { dim: 1 });
let w2 = Tensor::<2>::random([4096, 1024], Distribution::Default, &device).with_sharding(Split { dim: 0 });

let x = Tensor::<2>::random([8, 1024], Distribution::Default, &device);
// Split along its last dim, with nothing sent between members.
let hidden = activation::relu(x.matmul(w1));
// A partial sum of the members' shards, reduced when an op reads it whole.
let output = hidden.matmul(w2);
```

A sharded compute device's members are Cube devices, of one runtime or of several, or Flex devices. Members on
different runtimes form a sharded compute device whose every collective is emulated through the host: correct, and
far slower than members that share a runtime. A tensor made from data stays on the host until it
is used, and a split of it there, by `with_sharding` or by an op that consumes it, uploads each
member only its chunk. A random tensor waits the same way until it is first used, and a split use
has each member generate only its shard. Its random key is reserved when the tensor is created;
each shard receives a child key from `RandomKey::split`, so first-use order, member scheduling,
and later reseeding do not change its values. A split draw can differ from one device's with the
same seed. Any other tensor, `zeros` among them, is created whole on every member before `with_sharding` splits it.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
