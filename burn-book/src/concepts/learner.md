# Learner

The `Learner` is the main `struct` that let you train a neural network with support for `logging`,
`metric`, `checkpointing` and more. In order to create a learner, you must use the `LearnerBuilder`.

```rust,ignore
use burn::train::LearnerBuilder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::record::DefaultRecordSettings;

fn main() {
    let dataloader_train = ...;
    let dataloader_valid = ...;

    let model = ...;
    let optim = ...;

    let learner = LearnerBuilder::new("/tmp/artifact_dir")
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .with_file_checkpointer::<DefaultRecordSettings>(2)
        .num_epochs(10)
        .build(model, optim);

    let _model_trained = learner.fit(dataloader_train, dataloader_valid);
}
```

See this [example](https://github.com/burn-rs/burn/tree/main/examples/mnist) for a real usage.