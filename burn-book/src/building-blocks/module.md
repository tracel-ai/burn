# Module

The `Module` derive allows you to create your own neural network modules, similar to PyTorch.
The derive function only generates the necessary methods to essentially act as a parameter container for your type, it makes no assumptions about how the forward pass is declared.

```rust, ignore
use burn::nn;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: GELU,
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

If you want to create your own module that contains tensors, and not just other modules defined with the `Module` derive, you need to be careful to achieve the behavior you want.

- `Param<Tensor<B, D>>`:
If you want the tensor to be included as a parameter of your modules, you need to wrap the tensor in a `Param` struct.
This will create an ID that will be used to identify this parameter.
This is essential when performing module optimization and when saving states such as optimizer and module checkpoints.
Note that a module's record only contains parameters.

- `Param<Tensor<B, D>>.set_require_grad(false)`:
If you want the tensor to be included as a parameter of your modules, and therefore saved with the module's weights, but you don't want it to be updated by the optimizer.

- `Tensor<B, D>`:
If you want the tensor to act as a constant that can be recreated when instantiating a module.
This can be useful when generating sinusoidal embeddings, for example.


## Methods

These methods are available for all modules.

| Burn API                                               | PyTorch Equivalent                                      |
|--------------------------------------------------------|---------------------------------------------------------|
| `module.devices()`                                     | N/A                                                     |
| `module.fork(device)`                                  | Similar to `module.to(device).detach()`                 |
| `module.to_device(device)`                             | `module.to(device)`                                     |
| `module.no_grad()`                                     | `module.require_grad_(False)`                           |
| `module.num_params()`                                  | N/A                                                     |
| `module.visit(visitor)`                                | N/A                                                     |
| `module.map(mapper)`                                   | N/A                                                     |
| `module.into_record()`                                 | Similar to `state_dict`                                 |
| `module.load_record(record)`                           | Similar to `load_state_dict(state_dict)`                |
| `module.save_file(file_path, recorder)`                | N/A                                                     |
| `module.load_file(file_path, recorder)`                | N/A                                                     |


Similar to the backend trait, there is also the `ADModule` trait to signify a module with autodiff support.

| Burn API                                               | PyTorch Equivalent                                      |
|--------------------------------------------------------|---------------------------------------------------------|
| `module.valid()`                                       | `module.eval()`                                         |

## Visitor & Mapper

As mentioned earlier, modules primarily function as parameter containers.
Therefore, we naturally offer several ways to perform functions on each parameter.
This is distinct from PyTorch, where extending module functionalities is not as straightforward.

The `map` and `visitor` methods are quite similar but serve different purposes.
Mapping is used for potentially mutable operations where each parameter of a module can be updated to a new value.
In Burn, optimizers are essentially just sophisticated module mappers.
Visitors, on the other hand, are used when you don't intend to modify the module but need to retrieve specific information from it, such as the number of parameters or a list of devices in use.

You can implement your own mapper or visitor by implementing these simple traits:

```rust, ignore
/// Module visitor trait.
pub trait ModuleVisitor<B: Backend> {
    /// Visit a tensor in the module.
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>);
}

/// Module mapper trait.
pub trait ModuleMapper<B: Backend> {
    /// Map a tensor in the module.
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
}
```
