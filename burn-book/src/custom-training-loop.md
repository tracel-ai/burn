# Custom Training Loops

Even though Burn comes with a project dedicated to simplifying training, it doesn't mean that you
have to use it. Sometimes you may have special needs for your training, and it might be faster to
just reimplement the training loop yourself. Also, you may just prefer implementing your own
training loop instead of using a pre-built one in general.

Burn's got you covered!

We will start from the same example shown in the [basic workflow](./basic-workflow) section, but
without using the `Learner` struct.

```rust, ignore
#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    // Create the model and optimizer.
    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    // Create the batcher.
    let batcher = MnistBatcher::default();

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    ...
}
```

As seen with the previous example, setting up the configurations and the dataloader hasn't changed.
Now, let's move forward and write our own training loop:

```rust, ignore
pub fn run<B: AutodiffBackend>(device: B::Device) {
    ...

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.images);
            let loss = CrossEntropyLoss::new(None, &output.device())
                .forward(output.clone(), batch.targets.clone());
            let accuracy = accuracy(output, batch.targets);

            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {} | Accuracy {}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
                accuracy,
            );
        }
    }
}
```

In the previous code snippet, we can observe that the loop starts from epoch `1` and goes up to
`num_epochs`. Within each epoch, we iterate over the training dataloader. During this process, we
execute the forward pass, which is necessary for computing both the loss and accuracy. To maintain
simplicity, we print the results to stdout.

Upon obtaining the loss, we can invoke the `backward()` function, which returns the gradients
specific to each variable. It's important to note that we need to map these gradients to their
corresponding parameters using the `GradientsParams` type. This step is essential because you might
run multiple different autodiff graphs and accumulate gradients for each parameter id.

Finally, we can perform the optimization step using the learning rate, the model, and the computed
gradients. It's worth mentioning that, unlike PyTorch, there's no need to register the gradients
with the optimizer, nor do you have to call `zero_grad`. The gradients are automatically consumed
during the optimization step. If you're interested in gradient accumulation, you can easily achieve
this by using the `GradientsAccumulator`.

```rust, ignore
let mut accumulator = GradientsAccumulator::new();
let grads = model.backward();
let grads = GradientsParams::from_grads(grads, &model);
accumulator.accumulate(&model, grads); ...
let grads = accumulator.grads(); // Pop the accumulated gradients.
```

Note that after each epoch, we include a validation loop to assess our model's performance on
previously unseen data. To disable gradient tracking during this validation step, we can invoke
`model.valid()`, which provides a model on the inner backend without autodiff capabilities. It's
important to emphasize that we've declared our validation batcher to be on the inner backend,
specifically `MnistBatcher<B::InnerBackend>`; not using `model.valid()` will result in a compilation
error.

You can find the code above available as an
[example](https://github.com/tracel-ai/burn/tree/main/examples/custom-training-loop) for you to
test.

## Multiple optimizers

It's common practice to set different learning rates, optimizer parameters, or use different optimizers entirely, for different parts
of a model. In Burn, each `GradientParams` can contain only a subset of gradients to actually apply with an optimizer.
This allows you to flexibly mix and match optimizers!

```rust,ignore
// Start with calculating all gradients
let grads = loss.backward();

// Now split the gradients into various parts.
let grads_conv1 = GradientParams::from_module(&mut grads, &model.conv1);
let grads_conv2 = GradientParams::from_module(&mut grads, &model.conv2);

// You can step the model with these gradients, using different learning
// rates for each param. You could also use an entirely different optimizer here!
model = optim.step(config.lr * 2.0, model, grads_conv1);
model = optim.step(config.lr * 4.0, model, grads_conv2);

// For even more granular control you can split off individual parameter
// eg. a linear bias usually needs a smaller learning rate.
if let Some(bias) == model.linear1.bias {
    let grads_bias = GradientParams::from_params(&mut grads, &model.linear1, &[bias.id]);
    model = optim.step(config.lr * 0.1, model, grads_bias);
}

// Note that above calls remove gradients, so we can just get all "remaining" gradients.
let grads = GradientsParams::from_grads(grads, &model);
model = optim.step(config.lr, model, grads);
```

## Custom Type

The explanations above demonstrate how to create a basic training loop. However, you may find it
beneficial to organize your program using intermediary types. There are various ways to do this, but
it requires getting comfortable with generics.

If you wish to group the optimizer and the model into the same structure, you have several options.
It's important to note that the optimizer trait depends on both the `AutodiffModule` trait and the
`AutodiffBackend` trait, while the module only depends on the `AutodiffBackend` trait.

Here's a closer look at how you can create your types:

**Create a struct that is generic over the backend and the optimizer, with a predefined model.**

```rust, ignore
struct Learner<B, O>
where
    B: AutodiffBackend,
{
    model: Model<B>,
    optim: O,
}
```

This is quite straightforward. You can be generic over the backend since it's used with the concrete
type `Model` in this case.

**Create a struct that is generic over the model and the optimizer.**

```rust, ignore
struct Learner<M, O> {
    model: M,
    optim: O,
}
```

This option is a quite intuitive way to declare the struct. You don't need to write type constraints
with a `where` statement when defining a struct; you can wait until you implement the actual
function. However, with this struct, you may encounter some issues when trying to implement code
blocks to your struct.

```rust, ignore
impl<B, M, O> Learner<M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    pub fn step(&mut self, _batch: MnistBatch<B>) {
        //
    }
}
```

This will result in the following compilation error:

```console
1. the type parameter `B` is not constrained by the impl trait, self type, or predicates
   unconstrained type parameter [E0207]
```

To resolve this issue, you have two options. The first one is to make your function generic over
the backend and add your trait constraint within its definition:

```rust, ignore
#[allow(dead_code)]
impl<M, O> Learner2<M, O> {
    pub fn step<B: AutodiffBackend>(&mut self, _batch: MnistBatch<B>)
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
        O: Optimizer<M, B>,
    {
        //
    }
}
```

However, some people may prefer to have the constraints on the implementation block itself. In that
case, you can make your struct generic over the backend using `PhantomData<B>`.

**Create a struct that is generic over the backend, the model, and the optimizer.**

```rust, ignore
struct Learner3<B, M, O> {
    model: M,
    optim: O,
    _b: PhantomData<B>,
}
```

You might wonder why `PhantomData` is required. Each generic argument must be used as a field when
declaring a struct. When you don't need the generic argument, you can use `PhantomData` to mark it
as a zero sized type.

These are just some suggestions on how to define your own types, but you are free to use any pattern
that you prefer.
