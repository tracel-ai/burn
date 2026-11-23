mod common;

use std::sync::Arc;

use burn_core::tensor::{Device, TensorReadError};
use burn_std::ExecutionError;
use burn_train::{
    EventProcessorFailure, OnOptimizerStep, SupervisedTraining, TrainingError,
    logger::{FileMetricLogger, MetricLogger},
    metric::{
        Adaptor, LearningRateMetric, LossMetric, Metric, MetricAttributes, MetricMetadata,
        MetricName, Numeric, NumericAttributes, NumericEntry, OptimizerStepOutput, SerializedEntry,
        store::Split,
    },
};
use common::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    Update,
    Final,
}

struct ScalarInput(f64);

impl Adaptor<ScalarInput> for OptimizerStepOutput {
    fn adapt(&self) -> ScalarInput {
        ScalarInput(self.learning_rate().base())
    }
}

#[derive(Clone)]
struct ScalarMetric {
    name: MetricName,
    last: Option<f64>,
    failure: Failure,
}

impl ScalarMetric {
    fn new(name: &str, failure: Failure) -> Self {
        Self {
            name: Arc::new(name.into()),
            last: None,
            failure,
        }
    }
}

impl Metric for ScalarMetric {
    type Input = ScalarInput;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes::default().into()
    }

    fn clear(&mut self) {
        self.last = None;
    }

    fn update(
        &mut self,
        input: &ScalarInput,
        _: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        if self.failure == Failure::Update {
            return Err(ExecutionError::with_context("metric update failed").into());
        }
        let value = input.0;
        self.last = Some(value);
        Ok(SerializedEntry::new(
            value.to_string(),
            NumericEntry::Value(value).serialize(),
        ))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        if self.failure == Failure::Final {
            return Err(ExecutionError::with_context("metric finalization failed").into());
        }
        let value = self
            .last
            .ok_or_else(|| ExecutionError::with_context("no optimizer updates"))?;
        Ok(SerializedEntry::new(
            value.to_string(),
            NumericEntry::Final(value).serialize(),
        ))
    }
}

impl Numeric for ScalarMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.last.map(NumericEntry::Value)
    }

    fn running_value(&self) -> Option<NumericEntry> {
        None
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.last.unwrap())
    }
}

#[derive(Clone)]
struct StatusMetric;

impl Metric for StatusMetric {
    type Input = ();

    fn name(&self) -> MetricName {
        Arc::new("Optimizer Status".into())
    }

    fn clear(&mut self) {}

    fn update(
        &mut self,
        _: &(),
        metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let step = metadata.iteration.unwrap();
        assert_eq!(metadata.progress.items_processed, step);
        let status = format!("active {step}");
        Ok(SerializedEntry::new(status.clone(), status))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(SerializedEntry::not_available(None))
    }
}

#[test]
fn text_optimizer_metrics_follow_updates_and_reset_each_epoch() {
    let device = Device::flex().autodiff();
    let (train, valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .metrics_text((
            LossMetric::new(),
            OnOptimizerStep::with((StatusMetric, ScalarMetric::new("Custom LR", Failure::None))),
        ))
        .metrics(LearningRateMetric::new())
        .num_epochs(2)
        .grads_accumulation(2)
        .summary()
        .with_application_logger(None)
        .launch(make_learner(&device));
    assert!(result.error.is_none());
    for epoch in 1..=2 {
        assert_eq!(
            std::fs::read_to_string(
                dir.path()
                    .join(format!("train/epoch-{epoch}/Optimizer_Status.log"))
            )
            .unwrap(),
            "active 1\n"
        );
    }
    let mut logger = FileMetricLogger::new(dir.path());
    for split in [Split::Train, Split::Valid] {
        assert_eq!(logger.read_numeric("Loss", 1, &split).unwrap().len(), 3);
    }
    assert_eq!(
        logger
            .read_numeric("Custom LR", 1, &Split::Train)
            .unwrap()
            .len(),
        2
    );
    for name in ["Optimizer Status", "Custom LR"] {
        assert!(
            !dir.path()
                .join(format!("valid/epoch-1/{}.log", name.replace(' ', "_")))
                .exists()
        );
    }
}

#[rstest::rstest]
#[case(Failure::None)]
#[case(Failure::Update)]
#[case(Failure::Final)]
fn numeric_optimizer_metrics_adapt_inputs_and_propagate_failures(#[case] failure: Failure) {
    let device = Device::flex().autodiff();
    let (train, valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .metrics((
            LossMetric::new(),
            OnOptimizerStep::with((
                ScalarMetric::new("Fallible LR", failure),
                ScalarMetric::new("Custom LR", Failure::None),
            )),
        ))
        .with_application_logger(None)
        .launch(make_learner(&device));
    assert!(result.interrupted.is_none());
    match failure {
        Failure::None => assert!(result.error.is_none()),
        _ => {
            let Some(TrainingError::EventProcessor(error)) = result.error.as_deref() else {
                panic!("expected metric failure");
            };
            let EventProcessorFailure::Metric(error) = &error.failures()[0] else {
                panic!("wrong failure kind");
            };
            assert_eq!(error.metric.as_str(), "Fallible LR");
            assert_eq!(error.split, Split::Train);
        }
    }
    let mut logger = FileMetricLogger::new(dir.path());
    // A failing metric must not suppress later metrics or split finalization.
    let rates = logger.read_numeric("Custom LR", 1, &Split::Train).unwrap();
    assert!(matches!(rates.first(), Some(NumericEntry::Value(0.001))));
    assert!(matches!(rates.last(), Some(NumericEntry::Final(0.001))));
    let fallible_rates = logger
        .read_numeric("Fallible LR", 1, &Split::Train)
        .unwrap();
    match failure {
        Failure::None => assert!(matches!(
            fallible_rates.as_slice(),
            [
                NumericEntry::Value(0.001),
                NumericEntry::Value(0.001),
                NumericEntry::Final(0.001)
            ]
        )),
        Failure::Final => assert!(matches!(
            fallible_rates.as_slice(),
            [NumericEntry::Value(0.001), NumericEntry::Value(0.001)]
        )),
        Failure::Update => assert!(fallible_rates.is_empty()),
    }
}
