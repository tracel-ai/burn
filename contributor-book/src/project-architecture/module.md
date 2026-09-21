# Module

Modules organize parameters into structures that can be optimized, saved, and loaded.
`#[derive(Module)]` generates parameter traversal and training/validation conversions. A module does
not force the declaration of the forward pass, leaving it up to the implementer to decide how it
should be defined.

Configuration describes a module's structure and hyperparameters; records store its parameters
separately.

## Parameters and traversal

`Param<T>` gives a value an identity and supports lazy initialization. Tensor parameters use that
identity to associate optimizer state and gradients. `Param<Flag>` represents module-owned control
state, such as whether dropout or batch normalization behaves as during training.

`Module::visit` inspects parameters; `Module::map` transforms them. Visitors and mappers have hooks
for float, integer, and boolean tensors, control flags, and module paths. Reparameterizations such
as LoRA have nested parameters that participate in these traversals. `param.base()` reads the stored
base; `param.val()` materializes the effective value, including a reparameterization.

## Training and validation

`Module` includes the `valid(&self)` and `train(self)` transition hooks; the derive generates both
alongside traversal. Training and validation use the same module type, and a module can contain
parameters with different runtime autodiff contexts.

- `valid(&self)` creates a validation snapshot with autodiff and training flags disabled. It keeps
  configured trainability and flag settings, folds reparameterizations into parameter values, and
  removes checkpointing strategies with the autodiff association.
- `train(self)` enables autodiff and applies configured trainability and flags. It does not undo
  explicit freezing, reconstruct folded adapters, or restore discarded checkpointing strategies.
- `no_grad()` persistently disables parameter gradients while leaving control flags unchanged.
- `freeze()` and `unfreeze()` configure both gradients and flags; group variants target subtrees.

Keep the training module and use its `valid()` snapshot for validation. Inspect individual tensors
with `is_autodiff()`, `is_tracked()`, and `is_require_grad()` rather than inferring training state
from the trait or `module.devices()`. Device equality ignores autodiff settings, and the latter
method deduplicates compute resources.

## Optimization

[`Optimizer`](https://github.com/tracel-ai/burn/blob/main/crates/burn-optim/src/optim/module/base.rs)
updates one tensor at a time from its gradient and optional state. `State<D>` implements `Clone` and
`RecordState`, allowing tensors and scalars to be serialized independently of backend types.

[`ModuleOptimizer`](https://github.com/tracel-ai/burn/blob/main/crates/burn-optim/src/optim/module/module_optimizer.rs)
wraps these optimizers and manages module traversal, parameter groups, gradient lookup, device
migration, and per-parameter state. Parameter groups can use different optimizers.

An update proceeds as follows:

1. Run the model and call `loss.backward()`.
2. Convert the tensor gradients with `GradientsParams::from_grads(grads, &model)`.
3. Call `optimizer.step(learning_rate, model, grads)` to obtain the updated module.

The optimizer performs updates outside the autodiff graph and restores parameter trainability and
checkpointing strategy afterward. A transferred tracked parameter is an intermediate, so use
`model.fork(&device)` when the destination module should have independently optimizable leaves. Both
`to_device` and `fork` preserve source autodiff context; use `train()` to enable it explicitly.

See the [serialization chapter](./serialization.md) for `ModuleRecord` and `OptimizerRecord`.
