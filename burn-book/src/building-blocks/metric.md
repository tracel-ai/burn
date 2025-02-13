# Metric

When working with the learner, you have the option to record metrics that will be monitored
throughout the training process. We currently offer a restricted range of metrics.

| Metric           | Description                                             |
| ---------------- | ------------------------------------------------------- |
| Accuracy         | Calculate the accuracy in percentage                    |
| TopKAccuracy     | Calculate the top-k accuracy in percentage              |
| Precision        | Calculate precision in percentage                       |
| Recall           | Calculate recall in percentage                          |
| FBetaScore       | Calculate F<sub>β </sub>score in percentage             |
| AUROC            | Calculate the area under curve of ROC in percentage     |
| Loss             | Output the loss used for the backward pass              |
| CPU Temperature  | Fetch the temperature of CPUs                           |
| CPU Usage        | Fetch the CPU utilization                               |
| CPU Memory Usage | Fetch the CPU RAM usage                                 |
| GPU Temperature  | Fetch the GPU temperature                               |
| Learning Rate    | Fetch the current learning rate for each optimizer step |
| CUDA             | Fetch general CUDA metrics such as utilization          |

In order to use a metric, the output of your training step has to implement the `Adaptor` trait from
`burn-train::metric`. Here is an example for the classification output, already provided with the
crate.

```rust , ignore
/// Simple classification output adapted for multiple metrics.
#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 2>,

    /// The targets.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
```

# Custom Metric

Generating your own custom metrics is done by implementing the `Metric` trait.

```rust , ignore

/// Metric trait.
///
/// # Notes
///
/// Implementations should define their own input type only used by the metric.
/// This is important since some conflict may happen when the model output is adapted for each
/// metric's input type.
pub trait Metric: Send + Sync {
    /// The input type of the metric.
    type Input;

    /// The parametrized name of the metric.
    ///
    /// This should be unique, so avoid using short generic names, prefer using the long name.
    ///
    /// For a metric that can exist at different parameters (e.g., top-k accuracy for different
    /// values of k), the name should be unique for each instance.
    fn name(&self) -> String;

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry;
    /// Clear the metric state.
    fn clear(&mut self);
}
```

As an example, let's see how the loss metric is implemented.

```rust, ignore
/// The loss metric.
#[derive(Default)]
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    _b: B,
}

/// The loss metric input type.
#[derive(new)]
pub struct LossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}


impl<B: Backend> Metric for LossMetric<B> {
    type Input = LossInput<B>;

    fn update(&mut self, loss: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size] = loss.tensor.dims();
        let loss = loss
            .tensor
            .clone()
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap();

        self.state.update(
            loss,
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> String {
        "Loss".to_string()
    }
}
```

When the metric you are implementing is numeric in nature, you may want to also implement the
`Numeric` trait. This will allow your metric to be plotted.

```rust, ignore
impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
```
