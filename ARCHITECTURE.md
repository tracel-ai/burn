# Architecture

This file documents most major architectural decisions with the reasoning behind them.

__Sections__

* [Module](#module)
  * [Optimization](#optimization)
  * [Serialization](#serialization)


## Module

Modules are a way of creating neural network structures that can be easily optimized, saved, and loaded with little to no boilerplate.
Unlike other frameworks, a module does not force the declaration of the forward pass, leaving it up to the implementer to decide how it should be defined.
Additionally, most modules are created using a (de)serializable configuration, which defines the structure of the module and its hyper-parameters.
Parameters and hyper-parameters are not serialized into the same file and both are normaly necessary to load a module for inference.

### Optimization

Optimization is normally done with gradient descent (or ascent for reinforcement learning), and it is important to provide an easy API for optimizing modules.

#### Constraints

1. __Users should be able to control what is optimized.__ 
    Modules can contain anything for maximum flexibility, but not everything needs to be optimized.
2. __Optimizers should have a serializable state that is updated during training.__
    Many optimizers keep track of previous gradients to implement some form of momentum.
    However, the state can be anything, not just tensors, allowing for easy implementation of any kind of optimizer.
3. __The learning rate can be updated during training.__
    Learning rate schedulers are often used during training and should be considered as a key aspect.

#### Solution

The solution to this problem comprises multiple parts.
Firstly, the `Optimizer` trait is quite similar to the `Module` trait in terms of saving and loading the state.
Please refer to the [serialization](#serialization) section for more details.

Secondly, two traits were created.
The `Optimizer` trait is general and relatively unopinionated, with a simple `step` method that takes a learning rate, a module, and the gradients.
The other trait, `SimpleOptimizer`, aims to provide an easier API for implementing new optimizers.
The goal is to allow implementations to avoid handling missing gradients, loading and exporting records, navigating the module parameter structure, handling tracked and untracked tensors, and other such tasks.

Thirdly, each tensor that will be optimized needs to be wrapped into a `Param` struct, which gives them an ID used for (de)serialization and to associate the state of the optimizer to each parameter.
The `Module` trait has two ways to navigate over parameters.
The first one is the `map` function, which returns `Self` and makes it easy to implement any transformation and mutate all parameters.
The second one is the `visit` function, which has a similar signature but does not mutate the parameter tensors.

__SimpleOptimizer__

The `SimpleOptimizer` has two major assumptions:

1. The state of the optimizer is linked to each parameter.
In other words, each parameter has its own optimizer state, decoupled from the other parameters.
2. The state of the optimizer implements `Record`, `Clone`, and has a `'static` lifetime.

The benefits of those assumptions materialize in simplicity with little loss in flexibility.
The state associative type is also generic over the dimension, making it extremely easy to include tensors in the state that share the same dimensionality as its parameter.

To wrap a simple optimizer into the more general `Optimizer` trait, the `OptimizerAdaptor` struct is used.

__OptimizerAdaptor__

The `OptimizerAdaptor` is a simple struct composed of a `SimpleOptimizer` and a hashmap with all records associated with each parameter ID.
When performing an optimization step, the adaptor handles the following:

1. Updates each parameter tensor in the given module using the `Module::map` function.
2. Checks if a gradient for the current tensor exists.
3. Makes sure that the gradient, the tensor, and the optimizer state associated with the current parameter are on the same device.
The device can be different if the state is loaded from disk to restart training.
4. Performs the simple optimizer step using the inner tensor since the operations done by the optimizer should not be tracked in the autodiff graph.
5. Updates the state for the current parameter and returns the updated tensor, making sure it's properly registered into the autodiff graph if gradients are maked as required.

Note that a parameter can still be updated by another process, as is the case with running metrics used in batch norm.
These tensors are still wrapped using the `Param` struct so that they are included in the module's state and given a proper parameter ID, but they are not registered in the autodiff graph.

### Serialization

An important aspect of a deep learning framework is the ability to save and load models from disk.
Despite appearing as a simple feature, it involves numerous constraints that require a proper solution.

#### Constraints

1. __Users should be able to declare the precision of the model to be saved, independent of the backend in use.__

    The modules should not be duplicated in RAM in another precision to support this.
    Conversion should be done lazily during (de)serialization.

2. __Users should be able to add any field to a module, even fields that are not serializable.__

    This can include constants, database connections, other module references, or any other information.
    Only parameters should be serialized since the structure of the module itself should be encapsulated with module configurations (hyper-parameters).

3. __Users should be able to declare the format in which the module should be saved.__

    This can involve saving to a compressed JSON file or directly to bytes in memory for `no-std` environments.

4. __Users should be able to create a module with its saved parameters without having to initialize the module first.__

    This will avoid unnecessary module initialization and tensor loading, resulting in reduced cold start when dealing with inference.

In addition to all of these constraints, the solution should be easy to use.

#### Solution

In order to be able to add any field to a module without requiring it to be (de)serializable, we decouple the module type from its state.
We create a new type for each module that only contains the parameters that need to be saved.
To generate that type automatically, the user must either declare which field is a parameter or a constant, or we assume that each field implements the module trait.

The second solution was chosen as it simplifies the code generation and reduces the size of the user API.
This means that the `Module` trait should be implemented by [primitives types](./burn-core/src/module/param/primitive.rs).
The following diagrams highlight the main types and traits used in the solution.

<div align="center">
<h4>Module Serialization Types</h4>
<img src="./assets/ModuleSerialization.png" width="700px"/>
<div align="left">

The way the types interact with each other is pretty straightforward.
First, a module can be converted into a record using `into_record()`.
Note that tensors can be cloned, but it won't actually copy any data; it will create another reference to the same data.

Then, you can call `record()`, which takes a `RecordSettings` as a parameter.
The function is automatically implemented for each record.
It calls `into_item::<RecordSettings>()` on the record, which creates a serializable item following the given settings.
Note that tensors implement record, and their item is just a wrapper struct that contains information about the precision in which the tensor should be saved or loaded.
No actual copy of the tensor is made until this point.
The tensor is converted to the `Data` struct and then converted into the specified precision only when `serialize()` or `deserialize()` are called, which makes the whole process lazy.

To recapitulate, the `Module` trait has an associated type that implements `Record`, which only contains the parameters of the model.
The `Record` trait has a generic associated type (GAT) that specifies a family of types that can be (de)serialized given any `RecordSettings`.
Records are therefore decoupled from the backend in use, and the saved items can be loaded on any backend with any precision, since the conversion is type-safe and done when `serialize()` and `deserialize()` are called.
All of the types are generated using simple derive macros without any conditional statements or complex syntax, as `Record` and `Module` are implemented for all primitive types.
This makes the code simple and easy to maintain.
In addition, you can extend the current system with your own `Recorder` and `RecordSettings` to control how your modules should be saved and loaded.

##### Pros

* All constraints are respected.
* The code is simple and easy to maintain, with very few conditional statements.
It is just recursive data structures, where all the complexity is handled by the framework in primitive implementations.
* The user API is simple and small, with only two derives (`Record` and `Module`) and no additional attributes.
* Users can create their own `Module` and `Record` primitive types, which gives them the flexibility to control how their data is serialized without having to fork the framework.

##### Cons

* There are more types, but most of them are automatically generated and single-purpose, so users don't need to interact with them for common use cases.
However, they can do so if necessary.
* When instantiating a new record manually, each field must be set to something, even if the type itself is `()`, which represents no value.
Since the code generation step uses associative types, it doesn't know that a field type is actually nothing.
Creating a record manually without using the generated function `into_record` or loading it from a file is only useful to load a set of parameters into a module from an arbitrary source.
Using the record may not be the optimal solution to this problem, and another API could be created in the future.

##### Compatibility

Record may become incompatible with previous versions of Burn, depending on the chosen format.
The more compact format (bincode) store minimal information about the type, making it significantly smaller but less resilient to type changes such adding an optional field.
At some point, it might be necessary to provide a translation script that can translate a more resilient format from a previous version to a more compact one.
