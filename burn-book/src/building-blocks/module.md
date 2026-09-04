# Module

The `Module` derive allows you to create your own neural network modules, similar to PyTorch. The
derive function only generates the necessary methods to essentially act as a parameter container for
your type, it makes no assumptions about how the forward pass is declared.

```rust, ignore
use burn::module::Module;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward {
    linear_inner: Linear,
    linear_outer: Linear,
    dropout: Dropout,
    gelu: Gelu,
}

impl PositionWiseFeedForward {
    /// Normal method added to a struct.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

Note that all fields declared in the struct must also implement the `Module` trait.

## Forward Contract

The derive does not constrain `forward`, so a module's shape contract lives in two places: the doc
comment of the method, and assertions at its boundary.

### Documenting shapes

Modules in `burn::nn` document input and output shapes in a `# Shapes` section, one line per tensor,
with names for the axes that vary at runtime:

```rust, ignore
/// Applies the feed-forward block.
///
/// # Shapes
///
/// - input: `[batch_size, seq_length, d_model]`
/// - output: `[batch_size, seq_length, d_model]`
pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
```

Use the same convention for your own modules. It is the first thing a caller reads, and it is the
reference for the assertions below.

### Enforcing the contract

The rank of a `Tensor<D>` is a type parameter, so a caller cannot pass a `Tensor<4>` where a
`Tensor<3>` is expected, and `let [batch_size, seq_length, _] = input.dims();` names the axes with
the rank checked at compile time. The size of each axis is only known at runtime. Two macros,
available from the prelude, check those sizes at the boundary of a method:

- `assert_shape!` checks the axes of a tensor. It stays on in release builds.
- `debug_assert_shape!` is the same check, compiled out unless debug assertions are enabled.

Each takes a tensor and a bracketed pattern with one slot per axis. A slot is either `_`, which
skips the axis, `..`, which stands for any number of axes and may appear once, or any `usize`
expression the axis must equal:

```rust, ignore
let [batch_size, seq_length, _] = x.dims();
let flat = x.clone().reshape([batch_size * seq_length, 256]);

assert_shape!(x, [batch_size, seq_length, 256]);       // names from dims() and a literal
assert_shape!(flat, [batch_size * seq_length, 256]);   // arithmetic on names
assert_shape!(patch, [3 * 2, 2]);                      // arithmetic on literals
assert_shape!(mask, [batch_size, _]);                  // skip an axis
assert_shape!(hidden, [_, _, self.d_model]);           // a config field
assert_shape!(hidden, [.., self.d_model]);             // any rank, last axis checked
assert_shape!(image, [_, self.num_channels, ..]);      // channels first, any spatial rank
```

The pattern length is the expected rank. Since `tensor.dims()` returns `[usize; D]`, a pattern whose
length differs from `D` is a compile error rather than a runtime panic, and so is passing anything
other than a `Tensor`. That makes an exact pattern unusable in a method generic over
`const D: usize`, like the `PositionWiseFeedForward` at the top of this page. Use `..` there for the
axes you do not name; the pattern then checks a minimum rank at runtime instead:

```rust, ignore
pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
    assert_shape!(input, [.., self.d_model]);
    // ...
}
```

```rust, ignore
use burn::nn::{Gelu, Linear};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct FeedForward {
    linear_inner: Linear,
    linear_outer: Linear,
    gelu: Gelu,
    d_model: usize,
    d_ff: usize,
}

impl FeedForward {
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch_size, seq_length, _] = input.dims();
        assert_shape!(input, [_, _, self.d_model]);

        let x = self.linear_inner.forward(input);
        debug_assert_shape!(x, [batch_size, seq_length, self.d_ff]);

        let x = self.gelu.forward(x);
        let output = self.linear_outer.forward(x);

        assert_shape!(output, [batch_size, seq_length, self.d_model]);
        output
    }
}
```

The pattern reads like the `# Shapes` line above it. A failed check panics with the macro call, the
axis, the expected and actual sizes, and the full dims of the tensor:

```text
assert_shape!(output, [batch_size, seq_length, self.d_model]): axis 2 expected 512, got 2048 (dims [8, 128, 2048])
```

### Choosing a macro

- Name the runtime axes once at the top of `forward` with
  `let [batch_size, seq_length, _] = input.dims();` and use those names in every check.
- `assert_shape!` is the default for a contract check. A mismatch that reaches a tensor operation
  may fail there with a message about that operation's arguments, or broadcast silently when an axis
  has size 1. The check is a few integer comparisons, so its cost does not matter next to a kernel
  launch. Burn's own modules use it at their boundaries.
- `debug_assert_shape!` is for hot inner loops, or for internal invariants that an earlier always-on
  check already implies.

These checks complement the validation Burn performs inside each tensor operation. An operation
reports a mismatch in terms of its own arguments; a boundary check reports it in terms of the
module's contract, before any kernel runs.

These macros were inspired by the [`burn-contracts`](https://github.com/crutcher/burn-contracts)
crate by Crutcher Dunnavant.

## Tensor

If you want to create your own module that contains tensors, and not just other modules defined with
the `Module` derive, you need to be careful to achieve the behavior you want.

- `Param<Tensor<D>>`: If you want the tensor to be included as a parameter of your modules, you need
  to wrap the tensor in a `Param` struct. This will create an ID that will be used to identify this
  parameter. This is essential when performing module optimization and when saving states such as
  optimizer and module checkpoints. Note that a module's record only contains parameters.

- `Param<Tensor<D>>.set_require_grad(false)`: If you want the tensor to be included as a parameter
  of your modules, and therefore saved with the module's weights, but you don't want it to be
  updated by the optimizer.

- `Tensor<D>`: If you want the tensor to act as a constant that can be recreated when instantiating
  a module. This can be useful when generating sinusoidal embeddings, for example.

## Methods

These methods are available for all modules.

Gradient tracking and layer training behavior are separate. `no_grad` and `set_require_grad` change
only floating-point tensor parameters. `freeze` additionally disables module-owned training flags,
such as those controlling dropout and batch-normalization running statistics. Group variants apply
the same behavior only to values matched by the parameter group.

| Burn API                                           | PyTorch Equivalent                       |
| -------------------------------------------------- | ---------------------------------------- |
| `module.devices()`                                 | N/A                                      |
| `module.fork(device)`                              | Similar to `module.to(device).detach()`  |
| `module.to_device(device)`                         | `module.to(device)`                      |
| `module.set_require_grad(enabled)`                 | `module.requires_grad_(enabled)`         |
| `module.set_require_grad_group(group, enabled)`    | N/A                                      |
| `module.no_grad()`                                 | `module.requires_grad_(False)`           |
| `module.freeze()`                                  | N/A                                      |
| `module.unfreeze()`                                | N/A                                      |
| `module.num_params()`                              | N/A                                      |
| `module.visit(visitor)`                            | N/A                                      |
| `module.map(mapper)`                               | N/A                                      |
| `module.freeze_group(param_group)`                 | N/A                                      |
| `module.unfreeze_group(param_group)`               | N/A                                      |
| `module.apply_lora(lora)`                          | N/A                                      |
| `module.apply_qlora(qlora)`                        | N/A                                      |
| `module.apply_reparameterization(reparameterizer)` | N/A                                      |
| `module.into_record()`                             | Similar to `state_dict`                  |
| `module.load_record(record)`                       | Similar to `load_state_dict(state_dict)` |
| `module.try_load_record(record)`                   | Similar to `load_state_dict(state_dict)` |
| `module.try_load_file(file_path)`                  | Similar to `torch.load(...)`             |
| `module.load_file(file_path)`                      | Similar to `torch.load(...)`             |
| `module.save_file(file_path)`                      | Similar to `torch.save(state_dict, ...)` |

The `AutodiffModule` trait provides transitions between autodiff-enabled training modules and
inner-backend validation modules.

| Burn API         | PyTorch Equivalent |
| ---------------- | ------------------ |
| `module.valid()` | `module.eval()`    |
| `module.train()` | `module.train()`   |

Unlike their PyTorch counterparts, Burn's `valid()` and `train()` also transition a module between
its autodiff and inner backends. `valid()` temporarily disables gradient tracking and training
flags while preserving their configured state. `train()` returns the module to the autodiff backend
and reapplies that state; it does not undo an explicit `no_grad()` or `freeze()`.

Burn's `freeze()` and `unfreeze()` persistently set both tensor gradient tracking and module-owned
training flags, so they have no direct PyTorch equivalent.

## Visitor & Mapper

As mentioned earlier, modules primarily function as parameter containers. Therefore, we naturally
offer several ways to perform functions on each parameter. This is distinct from PyTorch, where
extending module functionalities is not as straightforward.

The `map` and `visitor` methods are quite similar but serve different purposes. Mapping is used for
potentially mutable operations where each parameter of a module can be updated to a new value. In
Burn, optimizers are essentially just sophisticated module mappers. Visitors, on the other hand, are
used when you don't intend to modify the module but need to retrieve specific information from it,
such as the number of parameters or a list of devices in use.

You can implement your own mapper or visitor by implementing these simple traits:

```rust, ignore
/// Module visitor trait.
pub trait ModuleVisitor {
    /// Visit a float tensor in the module.
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<D>);
    /// Visit an int tensor in the module.
    fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<D, Int>);
    /// Visit a bool tensor in the module.
    fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<D, Bool>);
}

/// Module mapper trait.
pub trait ModuleMapper {
    /// Map a float tensor in the module.
    fn map_float<const D: usize>(&mut self, id: ParamId, tensor: Tensor<D>) -> Tensor<D>;
    /// Map an int tensor in the module.
    fn map_int<const D: usize>(&mut self, id: ParamId, tensor: Tensor<D, Int>) -> Tensor<D, Int>;
    /// Map a bool tensor in the module.
    fn map_bool<const D: usize>(&mut self, id: ParamId, tensor: Tensor<D, Bool>) -> Tensor<D, Bool>;
}
```

Note that the trait doesn't require all methods to be implemented as they are already defined to
perform no operation. If you're only interested in float tensors (like the majority of use cases),
then you can simply implement `map_float` or `visit_float`.

For example, the `ModuleMapper` trait could be implemented to clamp all parameters into the range
`[min, max]`.

```rust, ignore
/// Clamp parameters into the range `[min, max]`.
pub struct Clamp {
    /// Lower-bound of the range.
    pub min: f32,
    /// Upper-bound of the range.
    pub max: f32,
}

// Clamp all floating-point parameter tensors between `[min, max]`.
impl ModuleMapper for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<D>,
    ) -> burn::prelude::Tensor<D> {
        tensor.clamp(self.min, self.max)
    }
}

// Clamp module mapper into the range `[-0.5, 0.5]`
let mut clamp = Clamp {
    min: -0.5,
    max: 0.5,
};
let model = model.map(&mut clamp);
```

If you want to use this during training to constrain your model parameters, make sure that the
parameter tensors are still tracked for autodiff. This can be done with a simple adjustment to the
implementation.

```rust, ignore
impl ModuleMapper for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<D>,
    ) -> burn::prelude::Tensor<D> {
        let is_require_grad = tensor.is_require_grad();

        let mut tensor = tensor.detach().clamp(self.min, self.max);

        if is_require_grad {
            tensor = tensor.require_grad();
        }

        tensor
    }
}
```

## Reparameterization

A reparameterization changes how a parameter's effective value is computed without changing the
module's type or forward pass. The original parameter remains its structural base, while an attached
state materializes the value returned by `Param::val()`. LoRA uses this mechanism to keep a frozen
structural base and attach trainable low-rank factors:

```rust, ignore
use burn::module::{Lora, Module};

let model = model.apply_lora(Lora::new(8, 16.0));
```

The `Reparameterizer` receives every floating-point parameter and its module path. It decides which
parameters to transform, prepares their structural bases, and optionally attaches a
`Reparameterization`. A reparameterization is itself a regular module, so its parameters
automatically participate in visitors, mappers, optimization, records, device transfers, and
autodiff. Custom techniques implement these traits and are applied through
`apply_reparameterization`, making custom parameter-level PEFT methods possible without modifying
the original model or layer.

LoRA and QLoRA are built on the same mechanism but provide the convenience methods `apply_lora` and
`apply_qlora` for normal use. Reparameterizations cannot currently be nested, so
`apply_reparameterization` should only be called on a module that does not already contain
reparameterized parameters. Use `Param::base()` to access the stored base directly and
`Param::val()` to obtain the materialized value.

## Module Display

Burn provides a simple way to display the structure of a module and its configuration at a glance.
You can print the module to see its structure, which is useful for debugging and tracking changes
across different versions of a module. (See the print output of the
[Basic Workflow Model](../basic-workflow/model.md) example.)

To customize the display of a module, you can implement the `ModuleDisplay` trait for your module.
This will change the default display settings for the module and its children. Note that
`ModuleDisplay` is automatically implemented for all modules, but you can override it to customize
the display by annotating the module with `#[module(custom_display)]`.

```rust
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PositionWiseFeedForward {
    linear_inner: Linear,
    linear_outer: Linear,
    dropout: Dropout,
    gelu: Gelu,
}

impl ModuleDisplay for PositionWiseFeedForward {
    /// Custom settings for the display of the module.
    /// If `None` is returned, the default settings will be used.
    fn custom_settings(&self) -> Option<burn::module::DisplaySettings> {
        DisplaySettings::new()
            // Will show all attributes (default is false)
            .with_show_all_attributes(false)
            // Will show each attribute on a new line (default is true)
            .with_new_line_after_attribute(true)
            // Will show the number of parameters (default is true)
            .with_show_num_parameters(true)
            // Will indent by 2 spaces (default is 2)
            .with_indentation_size(2)
            // Will show the parameter ID (default is false)
            .with_show_param_id(false)
            // Convenience method to wrap settings in Some()
            .optional()
    }

    /// Custom content to be displayed.
    /// If `None` is returned, the default content will be used
    /// (all attributes of the module)
    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("linear_inner", &self.linear_inner)
            .add("linear_outer", &self.linear_outer)
            .add("anything", "anything_else")
            .optional()
    }
}
```

## Built-in Modules

Burn comes with built-in modules that you can use to build your own modules.

### General

| Burn API            | PyTorch Equivalent                            |
| ------------------- | --------------------------------------------- |
| `BatchNorm`         | `nn.BatchNorm1d`, `nn.BatchNorm2d` etc.       |
| `Celu`              | `nn.CELU`                                     |
| `Dropout`           | `nn.Dropout`                                  |
| `Elu`               | `nn.ELU`                                      |
| `Embedding`         | `nn.Embedding`                                |
| `GaussianNoise`     | _No direct equivalent_                        |
| `Gelu`              | `nn.Gelu`                                     |
| `Glu`               | `nn.Glu`                                      |
| `GroupNorm`         | `nn.GroupNorm`                                |
| `HardShrink`        | `nn.Hardshrink`                               |
| `HardSigmoid`       | `nn.Hardsigmoid`                              |
| `CosineSimilarity`  | `nn.CosineSimilarity`                         |
| `HardSwish`         | `nn.Hardswish`                                |
| `InstanceNorm`      | `nn.InstanceNorm1d`, `nn.InstanceNorm2d` etc. |
| `LayerNorm`         | `nn.LayerNorm`                                |
| `LocalResponseNorm` | `nn.LocalResponseNorm`                        |
| `LeakyRelu`         | `nn.LeakyReLU`                                |
| `LogSigmoid`        | `nn.LogSigmoid`                               |
| `Mish`              | `nn.Mish`                                     |
| `Linear`            | `nn.Linear`                                   |
| `PairwiseDistance`  | `nn.PairwiseDistance`                         |
| `PixelShuffle`      | `nn.PixelShuffle`                             |
| `PixelUnshuffle`    | `nn.PixelUnshuffle`                           |
| `Prelu`             | `nn.PReLu`                                    |
| `Relu`              | `nn.ReLU`                                     |
| `Selu`              | `nn.SELU`                                     |
| `Sigmoid`           | `nn.Sigmoid`                                  |
| `SiLU`              | `nn.SiLU`                                     |
| `Softplus`          | `nn.Softplus`                                 |
| `SoftShrink`        | `nn.Softshrink`                               |
| `Softsign`          | `nn.Softsign`                                 |
| `Shrink`            | _No direct equivalent_                        |
| `RmsNorm`           | _No direct equivalent_                        |
| `SwiGlu`            | _No direct equivalent_                        |
| `Tanh`              | `nn.Tanh`                                     |
| `ThresholdedRelu`   | _No direct equivalent_                        |

### Convolutions

| Burn API          | PyTorch Equivalent             |
| ----------------- | ------------------------------ |
| `Conv1d`          | `nn.Conv1d`                    |
| `Conv2d`          | `nn.Conv2d`                    |
| `Conv3d`          | `nn.Conv3d`                    |
| `ConvTranspose1d` | `nn.ConvTranspose1d`           |
| `ConvTranspose2d` | `nn.ConvTranspose2d`           |
| `ConvTranspose3d` | `nn.ConvTranspose3d`           |
| `DeformConv2d`    | `torchvision.ops.DeformConv2d` |

### Pooling

| Burn API            | PyTorch Equivalent     |
| ------------------- | ---------------------- |
| `AdaptiveAvgPool1d` | `nn.AdaptiveAvgPool1d` |
| `AdaptiveAvgPool2d` | `nn.AdaptiveAvgPool2d` |
| `AvgPool1d`         | `nn.AvgPool1d`         |
| `AvgPool2d`         | `nn.AvgPool2d`         |
| `MaxPool1d`         | `nn.MaxPool1d`         |
| `MaxPool2d`         | `nn.MaxPool2d`         |

### Interpolation

| Burn API        | PyTorch Equivalent |
| --------------- | ------------------ |
| `Interpolate1d` | `nn.Upsample`      |
| `Interpolate2d` | `nn.Upsample`      |

Interpolation modules resize tensors using one of the available `InterpolateMode` options:

| Mode      | Description                                        |
| --------- | -------------------------------------------------- |
| `Nearest` | Nearest-neighbor interpolation                     |
| `Linear`  | Linear interpolation (bilinear for 2D)             |
| `Cubic`   | Cubic interpolation (bicubic for 2D)               |
| `Lanczos` | Lanczos3 resampling (6-tap sinc-based filter, a=3) |

Configuration is done via `Interpolate1dConfig` / `Interpolate2dConfig` with these options:

| Option          | Type                                   | Default   | Description                                             |
| --------------- | -------------------------------------- | --------- | ------------------------------------------------------- |
| `output_size`   | `Option<usize>` / `Option<[usize; 2]>` | `None`    | Target output size (takes precedence over scale_factor) |
| `scale_factor`  | `Option<f32>` / `Option<[f32; 2]>`     | `None`    | Scale factor for resizing                               |
| `mode`          | `InterpolateMode`                      | `Nearest` | Interpolation algorithm                                 |
| `align_corners` | `bool`                                 | `true`    | Align input/output corner pixels                        |

### RNNs

| Burn API         | PyTorch Equivalent     |
| ---------------- | ---------------------- |
| `Gru`/`BiGru`    | `nn.GRU`               |
| `Lstm`/`BiLstm`  | `nn.LSTM`              |
| `GateController` | _No direct equivalent_ |

### Transformer

| Burn API             | PyTorch Equivalent      |
| -------------------- | ----------------------- |
| `MultiHeadAttention` | `nn.MultiheadAttention` |
| `TransformerDecoder` | `nn.TransformerDecoder` |
| `TransformerEncoder` | `nn.TransformerEncoder` |
| `PositionalEncoding` | _No direct equivalent_  |
| `RotaryEncoding`     | _No direct equivalent_  |

### Loss

| Burn API                 | PyTorch Equivalent                |
| ------------------------ | --------------------------------- |
| `BinaryCrossEntropyLoss` | `nn.BCELoss`                      |
| `CosineEmbeddingLoss`    | `nn.CosineEmbeddingLoss`          |
| `CrossEntropyLoss`       | `nn.CrossEntropyLoss`             |
| `CTCLoss`                | `nn.CTCLoss`                      |
| `GramMatrixLoss`         | _No direct equivalent_            |
| `GaussianNLLLoss`        | `nn.GaussianNLLLoss`              |
| `HingeEmbeddingLoss`     | `nn.HingeEmbeddingLoss`           |
| `HuberLoss`              | `nn.HuberLoss`                    |
| `KLDivLoss`              | `nn.KLDivLoss`                    |
| `LpLoss`                 | _No direct equivalent_            |
| `MarginRankingLoss`      | `nn.MarginRankingLoss`            |
| `MseLoss`                | `nn.MSELoss`                      |
| `MultiMarginLoss`        | `nn.MultiMarginLoss`              |
| `PoissonNllLoss`         | `nn.PoissonNLLLoss`               |
| `RNNTLoss`               | `torchaudio.functional.rnnt_loss` |
| `SmoothL1Loss`           | `nn.SmoothL1Loss`                 |
| `TripletMarginLoss`      | `nn.TripletMarginLoss`            |
