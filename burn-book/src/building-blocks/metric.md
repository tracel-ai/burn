# Metric

When working with the learner, you have the option to record metrics that will be monitored
throughout the training process. We currently offer a restricted range of metrics.

| Metric              | Description                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Accuracy            | Calculate the accuracy in percentage                                                        |
| TopKAccuracy        | Calculate the top-k accuracy in percentage                                                  |
| Precision           | Calculate precision in percentage                                                           |
| Recall              | Calculate recall in percentage                                                              |
| FBetaScore          | Calculate F<sub>β </sub>score in percentage                                                 |
| AUROC               | Calculate the area under curve of ROC in percentage                                         |
| Loss                | Output the loss used for the backward pass                                                  |
| CharErrorRate (CER) | Calculate Character Error Rate in percentage                                                |
| WordErrorRate (WER) | Calculate Word Error Rate in percentage                                                     |
| HammingScore        | Calculate hamming score (also known as multi-label or label-based accuracy) in percentage   |
| Perplexity          | Calculate perplexity which is a measure of how well a probability model predicts samples    |
| IterationSpeed      | Tracks the training iteration speed, measuring how many iterations are completed per second |
| CPU Temperature     | Fetch the temperature of CPUs                                                               |
| CPU Usage           | Fetch the CPU utilization                                                                   |
| CPU Memory Usage    | Fetch the CPU RAM usage                                                                     |
| GPU Temperature     | Fetch the GPU temperature                                                                   |
| Learning Rate       | Fetch the current learning rate for each optimizer step                                     |
| CUDA                | Fetch general CUDA metrics such as utilization                                              |

| Vision Metric | Description                                                                              |
| ------------- | ---------------------------------------------------------------------------------------- |
| Dice          | Computes the Dice-Sorenson coefficient (DSC) for evaluating overlap between binary masks |
| PSNR          | Computes the peak signal-to-noise-ratio (PSNR) for image quality assessment              |
| SSIM          | Computes the structural similarity index measure (SSIM) for image quality assessment     |

## Using Metrics with the Learner

In order to use a metric, the output of your training step must implement the `Adaptor` trait from 
`burn-train::metric` for each metric's corresponding input type. The `Adaptor` trait simply converts 
your output struct into the input type the metric expects.

Burn provides four built-in output structs that cover common tasks. Each one already implements 
`Adaptor` for a set of metrics, so in many cases you can use them directly without writing any 
adaptor code yourself.

- `ClassificationOutput<B>`:
    - Use case: Single-label classification
    - Fields: `loss: Tensor<B, 1>`, `output: Tensor<B, 2>`, `targets: Tensor<B, 1, Int>`
    - Adapted metrics: Accuracy, TopKAccuracy, Perplexity, Precision\*, Recall\*, FBetaScore\*, AUROC\*, Loss
- `MultiLabelClassificationOutput<B>`:
    - Use case: Multi-label classification
    - Fields: `loss: Tensor<B, 1>`, `output: Tensor<B, 2>`, `targets: Tensor<B, 2, Int>`
    - Adapted metrics: HammingScore, Precision\*, Recall\*, FBetaScore\*, Loss
- `RegressionOutput<B>`:
    - Use case: Regression tasks
    - Fields: `loss: Tensor<B, 1>`, `output: Tensor<B, 2>`, `targets: Tensor<B, 2>`
    - Adapted metrics: Loss
- `SequenceOutput<B>`:
    - Use case: Sequence prediction
    - Fields: `loss: Tensor<B, 1>`, `output: Tensor<B, 2, Int>`, `targets: Tensor<B, 2, Int>`
    - Adapted metrics: Accuracy, TopKAccuracy, Perplexity, CER, WER, Loss

\* Precision, Recall, and FBetaScore all use `ConfusionStatsInput` as its input type so these three 
metrics are automatically (implicitly) adapted since `ConfusionStatsInput` is adapted.

If your metric isn't already adapted for the appropriate output struct, you can implement `Adaptor` yourself. 
For example, here is how `ClassificationOutput` adapts to `AccuracyInput`:

```rust,ignore
impl<B: Backend> Adaptor<AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}
```

If your task type is not covered by the built-in output structs, you can create an output struct for your data
and then adapt your metric for the output struct:

```rust,ignore
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
```

You can also open an issue on the [GitHub repository](https://github.com/tracel-ai/burn) when your task type is 
not covered by the built-in output structs. However, since creating an output struct for your data is simple, 
it is recommended to try creating your own output struct first. 

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
pub trait Metric: Send + Sync + Clone {
    /// The input type of the metric.
    type Input;

    /// The parameterized name of the metric.
    ///
    /// This should be unique, so avoid using short generic names, prefer using the long name.
    ///
    /// For a metric that can exist at different parameters (e.g., top-k accuracy for different
    /// values of k), the name should be unique for each instance.
    fn name(&self) -> MetricName;

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry;

    /// Clear the metric state.
    fn clear(&mut self);
}
```

As an example, let's see how the loss metric is implemented.

```rust, ignore
/// The loss metric.
#[derive(Clone)]
pub struct LossMetric<B: Backend> {
    name: Arc<String>,
    state: NumericMetricState,
    _b: B,
}

/// The [loss metric](LossMetric) input type.
#[derive(new)]
pub struct LossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}

impl<B: Backend> Default for LossMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> LossMetric<B> {
    /// Create the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Loss".to_string()),
            state: NumericMetricState::default(),
            _b: Default::default(),
        }
    }
}


impl<B: Backend> Metric for LossMetric<B> {
    type Input = LossInput<B>;

    fn update(&mut self, loss: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
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

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}
```

When the metric you are implementing is numeric in nature, you may want to also implement the
`Numeric` trait. This will allow your metric to be plotted.

```rust, ignore
impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
```
