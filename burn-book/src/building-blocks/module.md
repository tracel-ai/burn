# Module

The `Module` derive allows you to create your own neural network modules, similar to PyTorch. The
derive function only generates the necessary methods to essentially act as a parameter container for
your type, it makes no assumptions about how the forward pass is declared.

```rust, ignore
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    /// Normal method added to a struct.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

Note that all fields declared in the struct must also implement the `Module` trait.

## Tensor

If you want to create your own module that contains tensors, and not just other modules defined with
the `Module` derive, you need to be careful to achieve the behavior you want.

- `Param<Tensor<B, D>>`: If you want the tensor to be included as a parameter of your modules, you
  need to wrap the tensor in a `Param` struct. This will create an ID that will be used to identify
  this parameter. This is essential when performing module optimization and when saving states such
  as optimizer and module checkpoints. Note that a module's record only contains parameters.

- `Param<Tensor<B, D>>.set_require_grad(false)`: If you want the tensor to be included as a
  parameter of your modules, and therefore saved with the module's weights, but you don't want it to
  be updated by the optimizer.

- `Tensor<B, D>`: If you want the tensor to act as a constant that can be recreated when
  instantiating a module. This can be useful when generating sinusoidal embeddings, for example.

## Methods

These methods are available for all modules.

| Burn API                                | PyTorch Equivalent                       |
| --------------------------------------- | ---------------------------------------- |
| `module.devices()`                      | N/A                                      |
| `module.fork(device)`                   | Similar to `module.to(device).detach()`  |
| `module.to_device(device)`              | `module.to(device)`                      |
| `module.no_grad()`                      | `module.require_grad_(False)`            |
| `module.num_params()`                   | N/A                                      |
| `module.visit(visitor)`                 | N/A                                      |
| `module.map(mapper)`                    | N/A                                      |
| `module.into_record()`                  | Similar to `state_dict`                  |
| `module.load_record(record)`            | Similar to `load_state_dict(state_dict)` |
| `module.save_file(file_path, recorder)` | N/A                                      |
| `module.load_file(file_path, recorder)` | N/A                                      |

Similar to the backend trait, there is also the `AutodiffModule` trait to signify a module with
autodiff support.

| Burn API         | PyTorch Equivalent |
| ---------------- | ------------------ |
| `module.valid()` | `module.eval()`    |

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
pub trait ModuleVisitor<B: Backend> {
    /// Visit a float tensor in the module.
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>);
    /// Visit an int tensor in the module.
    fn visit_int<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Int>);
    /// Visit a bool tensor in the module.
    fn visit_bool<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D, Bool>);
}

/// Module mapper trait.
pub trait ModuleMapper<B: Backend> {
    /// Map a float tensor in the module.
    fn map_float<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
    /// Map an int tensor in the module.
    fn map_int<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D, Int>) -> Tensor<B, D, Int>;
    /// Map a bool tensor in the module.
    fn map_bool<const D: usize>(&mut self, id: ParamId, tensor: Tensor<B, D, Bool>) -> Tensor<B, D, Bool>;
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
impl<B: Backend> ModuleMapper<B> for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<B, D>,
    ) -> burn::prelude::Tensor<B, D> {
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
impl<B: AutodiffBackend> ModuleMapper<B> for Clamp {
    fn map_float<const D: usize>(
        &mut self,
        _id: burn::module::ParamId,
        tensor: burn::prelude::Tensor<B, D>,
    ) -> burn::prelude::Tensor<B, D> {
        let is_require_grad = tensor.is_require_grad();

        let mut tensor = Tensor::from_inner(tensor.inner().clamp(self.min, self.max));

        if is_require_grad {
            tensor = tensor.require_grad();
        }

        tensor
    }
}
```

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
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl<B: Backend> ModuleDisplay for PositionWiseFeedForward<B> {
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

| Burn API        | PyTorch Equivalent                            |
| --------------- | --------------------------------------------- |
| `BatchNorm`     | `nn.BatchNorm1d`, `nn.BatchNorm2d` etc.       |
| `Dropout`       | `nn.Dropout`                                  |
| `Embedding`     | `nn.Embedding`                                |
| `Gelu`          | `nn.Gelu`                                     |
| `GroupNorm`     | `nn.GroupNorm`                                |
| `HardSigmoid`   | `nn.Hardsigmoid`                              |
| `InstanceNorm`  | `nn.InstanceNorm1d`, `nn.InstanceNorm2d` etc. |
| `LayerNorm`     | `nn.LayerNorm`                                |
| `LeakyRelu`     | `nn.LeakyReLU`                                |
| `Linear`        | `nn.Linear`                                   |
| `Prelu`         | `nn.PReLu`                                    |
| `Relu`          | `nn.ReLU`                                     |
| `RmsNorm`       | _No direct equivalent_                        |
| `SwiGlu`        | _No direct equivalent_                        |
| `Interpolate1d` | _No direct equivalent_                        |
| `Interpolate2d` | _No direct equivalent_                        |

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

### RNNs

| Burn API         | PyTorch Equivalent     |
| ---------------- | ---------------------- |
| `Gru`            | `nn.GRU`               |
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

| Burn API           | PyTorch Equivalent    |
| ------------------ | --------------------- |
| `CrossEntropyLoss` | `nn.CrossEntropyLoss` |
| `MseLoss`          | `nn.MSELoss`          |
| `HuberLoss`        | `nn.HuberLoss`        |
| `PoissonNllLoss`   | `nn.PoissonNLLLoss`   |
