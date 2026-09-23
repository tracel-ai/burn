# Burn RL

<!-- This crate should be used with [burn](https://github.com/tracel-ai/burn). -->

<!-- [![Current Crates.io Version](https://img.shields.io/crates/v/burn-rl.svg)](https://crates.io/crates/burn-rl)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-rl/blob/master/README.md) -->

Reinforcement learning building blocks for [Burn](https://github.com/tracel-ai/burn).

## Usage

`burn-rl` is not a default dependency of `burn`. Enable the `rl` feature, together with `train` for
the RL learner:

```toml
burn = { version = "0.22", features = ["train", "rl", "flex"] }
```

The crate is re-exported as `burn::rl`. See the
[DQN agent example](https://github.com/tracel-ai/burn/tree/main/examples/dqn-agent).
