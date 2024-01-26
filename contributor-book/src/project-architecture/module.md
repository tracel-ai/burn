# Module

Modules are a way of creating neural network structures that can be easily optimized, saved, and loaded with little to no boilerplate.
Unlike other frameworks, a module does not force the declaration of the forward pass, leaving it up to the implementer to decide how it should be defined.

Additionally, most modules are created using a (de)serializable configuration, which defines the structure of the module and its hyper-parameters.
Parameters and hyper-parameters are not serialized into the same file and both are normally necessary to load a module for inference.

## Optimization

Optimization is normally done with gradient descent (or ascent for reinforcement learning), and it is important to provide an easy API for optimizing modules.

### Constraints

1. **Users should be able to control what is optimized.**
   Modules can contain anything for maximum flexibility, but not everything needs to be optimized.
2. **Optimizers should have a serializable state that is updated during training.**
   Many optimizers keep track of previous gradients to implement some form of momentum.
   However, the state can be anything, not just tensors, allowing for easy implementation of any kind of optimizer.
3. **The learning rate can be updated during training.**
   Learning rate schedulers are often used during training and should be considered as a key aspect.

### Solution

`Module` trait defined in [`burn-core/src/module/base.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-core/src/module/base.rs#L83)
`Optimizer` trait defined in [`burn-core/src/optim/base.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-core/src/optim/base.rs#L8)

The solution to this problem comprises multiple parts.
Firstly, the `Optimizer` trait is quite similar to the `Module` trait, in terms of saving and loading the state. Please refer to the [serialization](./serialization.md) section for more details.

Secondly, two traits were created.
The `Optimizer` trait is general and relatively unopinionated, with a simple `step` method that takes a learning rate, a module, and the gradients.
The other trait, `SimpleOptimizer`, aims to provide an easier API for implementing new optimizers.
The goal is to allow implementations to avoid handling missing gradients, loading and exporting records, navigating the module parameter structure, handling tracked and untracked tensors, and other such tasks.

Thirdly, each tensor that will be optimized needs to be wrapped into a `Param` struct, which gives them an ID used for (de)serialization and to associate the state of the optimizer to each parameter.
The `Module` trait has two ways to navigate over parameters.
The first one is the `map` function, which returns `Self` and makes it easy to implement any transformation and mutate all parameters.
The second one is the `visit` function, which has a similar signature but does not mutate the parameter tensors.

#### SimpleOptimizer

located in  [`burn-core/src/optim/simple/base.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-core/src/optim/simple/base.rs#L9)

The `SimpleOptimizer` has two major assumptions:

1. The state of the optimizer is linked to each parameter.
   In other words, each parameter has its own optimizer state, decoupled from the other parameters.
2. The state of the optimizer implements `Record`, `Clone`, and has a `'static` lifetime.

The benefits of those assumptions materialize in simplicity with little loss in flexibility.
The state associative type is also generic over the dimension, making it extremely easy to include tensors in the state that share the same dimensionality as its parameter.

To wrap a simple optimizer into the more general `Optimizer` trait, the `OptimizerAdaptor` struct is used.

#### OptimizerAdaptor

Located in in [`burn-core/src/optim/simple/adapter.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-core/src/optim/simple/adaptor.rs#L14)

The `OptimizerAdaptor` is a simple struct composed of a `SimpleOptimizer` and a hashmap with all records associated with each parameter ID.

When performing an optimization step, the adaptor handles the following:

1. Updates each parameter tensor in the given module using the `Module::map` function.
2. Checks if a gradient for the current tensor exists.
3. Makes sure that the gradient, the tensor, and the optimizer state associated with the current parameter are on the same device.
   The device can be different if the state is loaded from disk to restart training.
4. Performs the simple optimizer step using the inner tensor since the operations done by the optimizer should not be tracked in the autodiff graph.
5. Updates the state for the current parameter and returns the updated tensor, making sure it's properly registered into the autodiff graph if gradients are marked as required.

Note that a parameter can still be updated by another process, as is the case with running metrics used in batch norm.
These tensors are still wrapped using the `Param` struct so that they are included in the module's state and given a proper parameter ID, but they are not registered in the autodiff graph.
