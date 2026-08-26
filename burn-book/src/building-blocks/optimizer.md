# Optimizer

Optimizers update a module's trainable parameters from their gradients. Burn provides common
optimizers such as SGD, Adam, AdamW, AdaGrad, RMSProp, Adan, LAMB, and Muon in `burn-optim`,
re-exported under `burn::optim`.

Most applications interact with a [`ModuleOptimizer`](#moduleoptimizer). Create one from an
optimizer configuration, then pass it to a `Learner` or call `step` in a custom training loop:

```rust, ignore
use burn::optim::AdamConfig;

let optimizer = AdamConfig::new().init();
let learner = Learner::new(model, optimizer, learning_rate);
```

Configuration builders expose optimizer-specific options such as momentum, weight decay, AMSGrad,
and gradient clipping. For example:

```rust, ignore
use burn::optim::{AdamWConfig, grad_clipping::GradientClippingConfig};

let optimizer = AdamWConfig::new()
    .with_weight_decay(5e-5)
    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    .init();
```

## Custom Training Loop

In a custom loop, first run backpropagation and associate the tensor gradients with the module's
parameter IDs. `ModuleOptimizer::step` consumes those gradients, updates its state, and returns the
updated module:

```rust, ignore
use burn::optim::{AdamConfig, GradientsParams};

let mut optimizer = AdamConfig::new().init();

let output = model.forward(input);
let loss = loss_fn.forward(output, targets);
let gradients = loss.backward();
let gradients = GradientsParams::from_grads(gradients, &model);

model = optimizer.step(learning_rate.into(), model, gradients);
```

Unlike optimizers that store gradients on every parameter, Burn returns gradients from `backward`.
There is no separate `zero_grad` call: the gradient container is consumed by `step`. For gradient
accumulation, use `GradientsAccumulator` before calling the optimizer.

The first argument to `step` is a `ModuleLearningRate`. A single `f64` learning rate can be passed
to represent a `ModuleLearningRate`, while a module learning-rate scheduler produces grouped
learning rates directly. See [Learning Rate Scheduler](./lr-scheduler.md).

## `ModuleOptimizer`

The low-level [`Optimizer`](#implementing-an-optimizer) trait updates one tensor at a time.
`ModuleOptimizer` adapts it to an entire module and handles the surrounding mechanics:

1. It traverses trainable parameters through `Module::map`.
2. It finds and consumes each parameter's gradient by `ParamId`.
3. It chooses the optimizer and learning rate assigned to that parameter group.
4. It moves optimizer state to the parameter's device when necessary.
5. It calls the per-tensor optimizer and stores the returned state.
6. It preserves whether the updated parameter requires gradients.

`ModuleOptimizer` is intentionally non-generic over both the module and optimizer. This makes it
possible to store different optimizer implementations for different parameter groups while keeping
the training API and checkpoint type stable.

## Parameter Groups

`ParamGroup` selects parameters by ID or by their path in the module. Common constructors include:

| Constructor                               | Matches                             |
| ----------------------------------------- | ----------------------------------- |
| `ParamGroup::all()`                       | Every parameter                     |
| `ParamGroup::from_ids(ids)`               | Explicit parameter IDs              |
| `ParamGroup::from_path("encoder.weight")` | One exact module path               |
| `ParamGroup::from_predicate("encoder")`   | Paths containing the predicate      |
| `ParamGroup::from_regex(pattern)?`        | Paths matching a regular expression |

Groups can be combined and can exclude another group. For example, this selects an encoder except
for its biases:

```rust, ignore
use burn::module::ParamGroup;

let encoder = ParamGroup::from_predicate("encoder")
    .exclude(ParamGroup::from_predicate("bias"));
```

Add a group-specific optimizer with `ModuleOptimizer::with_group`:

```rust, ignore
let optimizer = default_optimizer.with_group(
    ParamGroup::from_predicate("encoder"),
    encoder_optimizer,
    None, // Optional gradient clipping for this group.
);
```

The initial optimizer is the fallback and matches every parameter. If several added groups match a
parameter, the last group takes precedence. Adding a group after optimization has started clears the
existing state of parameters matched by that group, because their state may belong to a different
optimizer type.

Learning-rate schedulers support the same grouping model, so optimizer choice and learning-rate
policy can be assigned independently. See [Learner](./learner.md#parameter-groups).

## Optimizer State and Records

State belongs to the optimizer and is stored per parameter. Adam, for example, records first- and
second-order momentum tensors and a step counter. `ModuleOptimizer` lazily creates this state on the
first update.

The state can be checkpointed as an `OptimizerRecord`:

```rust, ignore
optimizer.save("optimizer")?;
let optimizer = AdamConfig::new().init().load("optimizer")?;
```

An optimizer record is keyed by `ParamId`. It contains named tensor leaves, typed scalar leaves, the
tensor rank, and the parameter path used to restore parameter-group routing. State tensors are moved
to the corresponding parameter device on the next step, so loading does not require a device
argument. The `Learner` handles optimizer checkpoints automatically when checkpointers are enabled.

See [Record](./record.md) for the common save, load, and in-memory byte APIs.

## Implementing an Optimizer

Optimizer authors implement the per-tensor `Optimizer` trait. Its associated state is generic over
the parameter rank and implements `RecordState`:

```rust, ignore
use burn::{
    optim::{LearningRate, ModuleOptimizer, Optimizer, RecordState},
    tensor::{Device, Tensor},
};

#[derive(Clone)]
pub struct MySgd;

impl Optimizer for MySgd {
    type State<const D: usize> = ();

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        _state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        (tensor - grad.mul_scalar(lr), None)
    }

    fn to_device<const D: usize>(state: Self::State<D>, _device: &Device) -> Self::State<D> {
        state
    }
}

let optimizer = ModuleOptimizer::from(MySgd);
```

For a stateful optimizer, define a state structure and derive `RecordState`. The derive supports
tensors, optional tensors, vectors of tensors, scalars, and nested state structures:

```rust, ignore
#[derive(RecordState, Clone)]
pub struct MomentumState<const D: usize> {
    velocity: Tensor<D>,
    step: usize,
}
```

The `step` method receives the previous state, if any, and returns the updated tensor and optional
new state. `to_device` moves every tensor held by the state to the requested device.
