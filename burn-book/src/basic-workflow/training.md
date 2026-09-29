# Training

We are now ready to write the necessary code to train our model on the MNIST dataset. We shall
define the code for this training section in the file: `src/training.rs`.

Instead of a simple tensor, the model should output an item that can be understood by the learner, a
struct whose responsibility is to apply an optimizer to the model. The output struct is used for all
metrics calculated during the training. Therefore it should include all the necessary information to
calculate any metric that you want for a task.

Burn provides two basic output types: `ClassificationOutput` and `RegressionOutput`. They implement
the necessary trait to be used with metrics. It is possible to create your own item, but it is
beyond the scope of this guide.

Since the MNIST task is a classification problem, we will use the `ClassificationOutput` type.

```rust,ignore
{{#include ../../../examples/guide/src/training.rs:classification}}
```

As evident from the preceding code block, we employ the cross-entropy loss module for loss
calculation, without the inclusion of any padding token. We then return the classification output
containing the loss, the output tensor with all logits and the targets.

Please take note that tensor operations receive owned tensors as input. For reusing a tensor
multiple times, you need to use the `clone()` function. There's no need to worry; this process won't
involve actual copying of the tensor data. Instead, it will simply indicate that the tensor is
employed in multiple instances, implying that certain operations won't be performed in place. In
summary, our API has been designed with owned tensors to optimize performance.

Moving forward, we will proceed with the implementation of both the training and validation steps
for our model.

```rust,ignore
{{#include ../../../examples/guide/src/training.rs:steps}}
```

Here we define the `Input` and `Output` associated types of `TrainStep` and `InferenceStep` as
`MnistBatch` and `ClassificationOutput`. In the training step, the computation of gradients is
straightforward, necessitating a simple invocation of `backward()` on the loss. Note that contrary
to PyTorch, gradients are not stored alongside each tensor parameter, but are rather returned by the
backward pass, as such: `let gradients = loss.backward();`. The gradient of a parameter can be
obtained with the grad function: `let grad = tensor.grad(&gradients);`. Although it is not necessary
when using the learner struct and the optimizers, it can prove to be quite useful when debugging or
writing custom training loops. One difference between training and validation is the device mode:
training uses an autodiff-enabled device so `backward` records and traverses the graph.

<details>
<summary><strong>🦀 Associated Types in Trait Implementations</strong></summary>

Although generic data types, trait and trait bounds were already introduced in previous sections of
this guide, the previous code snippet might be a lot to take in at first.

In the example above, we implement the `TrainStep` and `InferenceStep` trait for our `Model` struct,
which contains runtime-dispatched tensors as covered before. These traits are provided by
`burn::train` and define a common `step` method that should be implemented for all structs. Since
traits declare associated types, each implementation specifies `type Input` and `type Output`. Here
those types are `MnistBatch` and `ClassificationOutput`. The `step` method signature uses these
concrete types.

For more details, see the
[associated types section](https://doc.rust-lang.org/book/ch20-02-advanced-traits.html#specifying-placeholder-types-in-trait-definitions-with-associated-types)
of the Rust Book.

</details><br>

Let us move on to establishing the practical training configuration.

```rust,ignore
{{#include ../../../examples/guide/src/training.rs:training}}
```

It is a good practice to use the `Config` derive to create the experiment configuration. In the
`train` function, the first thing we are doing is making sure the `artifact_dir` exists, using the
standard rust library for file manipulation. All checkpoints, logging and metrics will be stored
under this directory. We initialize the dataloaders using the previously created batcher. The
selected runtime device is cloned and switched to autodiff mode before the model is initialized.

Next, we create a supervised training runner with the dataloaders for training and validation and we
register the accuracy and loss metric on both training and validation steps. We also enable
checkpointing with `with_default_checkpointers()`, which periodically saves the model, optimizer,
and learning rate scheduler state to burnpack files under the experiment directory so training can
be resumed.

For the sake of simplicity in this example, we employ the test set as the validation set; however,
we do not recommend this practice for actual usage.

We create the learner containing the model, the optimizer and the learning rate. Notably, the third
argument of the learner's `new` function should actually be a learning rate _scheduler_. When
provided with a float as in our example, it is automatically transformed into a _constant_ learning
rate scheduler. The learning rate is not part of the optimizer config as it is often done in other
frameworks, but rather passed as a parameter when executing the optimizer step. This avoids having
to mutate the state of the optimizer and is therefore more functional. It makes no difference when
using the learner struct, but it will be an essential nuance to grasp if you implement your own
training loop.

Once the learner and supervised training instance are created, we can call `training.launch` and
provide the learner.

Finally, the trained model is returned by the `launch` method. The trained weights are then saved by
taking a record with `into_record()` and calling `save`, which writes a burnpack (`.bpk`) file. A
record holds plain tensor data, so any backend, regardless of precision, can load recorded weights
of any kind.
