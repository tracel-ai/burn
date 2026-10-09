# Burn RL

> Reinforcement learning building blocks for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-rl.svg)](https://crates.io/crates/burn-rl)
[![Documentation](https://docs.rs/burn-rl/badge.svg)](https://docs.rs/burn-rl)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

- `Environment`: a simulation an agent acts in, returning a `StepResult` per action.
- `Policy`: maps observations to actions; `PolicyLearner` updates a policy from experience.
- `TransitionBuffer`: a replay buffer of `Transition`s sampled in batches.

## Usage

`burn-rl` is not a default dependency of `burn`. Enable the `rl` feature, together with `train` for
the RL learner:

```toml
burn = { version = "0.22", features = ["train", "rl", "flex"] }
```

The crate is re-exported as `burn::rl`. See the
[DQN agent example](https://github.com/tracel-ai/burn/tree/main/examples/dqn-agent).

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
