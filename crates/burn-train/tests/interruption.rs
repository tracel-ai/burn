mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use burn_core::tensor::{Device, TensorReadError};
use burn_std::ExecutionError;
use burn_train::{
    EvaluatorBuilder, EventProcessorFailure, Interrupter, Interruption, LearningResult,
    SupervisedTraining, TrainingError,
    logger::InMemoryMetricLogger,
    metric::{Metric, MetricMetadata, MetricName, SerializedEntry, store::Split},
    train::WorkerFailure,
};
use common::*;

/// A metric that cannot read its input from the `fail_from`-th update on, the way a metric
/// fails when the computation it reads failed.
#[derive(Clone)]
struct FailingMetric {
    updates: Arc<AtomicUsize>,
    fail_from: usize,
}

impl Metric for FailingMetric {
    type Input = ();

    fn name(&self) -> MetricName {
        Arc::new("Failing".to_string())
    }

    fn update(
        &mut self,
        _item: &(),
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let update = self.updates.fetch_add(1, Ordering::SeqCst) + 1;
        if update >= self.fail_from {
            return Err(ExecutionError::with_context("the metric could not read its input").into());
        }
        Ok(SerializedEntry::new("ok".into(), "ok".into()))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(SerializedEntry::new("ok".into(), "ok".into()))
    }

    fn clear(&mut self) {}
}

fn train(fail_from: usize) -> (LearningResult<ToyModel>, usize) {
    let learner = make_learner(&Device::flex().autodiff());
    let (dl_train, dl_valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let updates = Arc::new(AtomicUsize::new(0));

    let result = SupervisedTraining::new(dir.path(), dl_train, dl_valid)
        .num_epochs(3)
        .metric_train(FailingMetric {
            updates: updates.clone(),
            fail_from,
        })
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);

    (result, updates.load(Ordering::SeqCst))
}

/// The failing metric's name and split, from the reported error.
fn metric_failure(error: Option<Arc<TrainingError>>) -> (String, Split) {
    match error.as_deref() {
        Some(TrainingError::EventProcessor(err)) => match &err.failures()[0] {
            EventProcessorFailure::Metric(failure) => {
                (failure.metric.to_string(), failure.split.clone())
            }
            other => panic!("expected a metric failure, got {other}"),
        },
        other => panic!("expected a metric error, got {other:?}"),
    }
}

#[test]
fn a_metric_error_stops_training_and_is_reported_as_an_error() {
    let (result, updates) = train(1);

    assert_eq!(result.interrupted, None, "an error is not an interruption");
    let (metric, split) = metric_failure(result.error);
    assert_eq!(metric, "Failing");
    assert_eq!(split, Split::Train);
    // Stopped in the first epoch, instead of running all three. The asynchronous processor
    // reports an item's error on a later call, so a couple of items can go through first.
    assert!(
        updates < 6,
        "training went on after the error: {updates} updates"
    );
}

#[test]
fn a_completed_training_reports_neither() {
    let (result, _) = train(usize::MAX);

    assert_eq!(result.interrupted, None);
    assert!(result.error.is_none());
}

#[test]
fn a_stop_request_and_an_error_are_kept_apart() {
    let interrupter = Interrupter::new();
    assert!(!interrupter.should_stop());

    interrupter.stop(Some("the user asked"));
    assert!(interrupter.should_stop());
    assert_eq!(
        interrupter.interruption(),
        Some(Interruption {
            reason: Some("the user asked".to_string())
        })
    );
    assert!(interrupter.error().is_none());

    let failed = Interrupter::new();
    failed.fail_on_error(Ok::<(), TrainingError>(()));
    assert!(!failed.should_stop());

    let worker = |message: &str| {
        TrainingError::Workers(vec![WorkerFailure {
            device_id: 0,
            message: message.to_string(),
        }])
    };
    failed.fail(worker("the first error"));
    failed.fail(worker("a consequence of it"));
    assert!(failed.should_stop());
    assert_eq!(
        failed.interruption(),
        None,
        "an error is not an interruption"
    );
    match failed.error().as_deref() {
        Some(TrainingError::Workers(failures)) => {
            assert_eq!(failures[0].message, "the first error")
        }
        other => panic!("expected the first error, got {other:?}"),
    }

    // A stop requested once an error happened still counts as that error.
    failed.stop(Some("the user asked too"));
    assert_eq!(failed.interruption(), None);
    assert!(failed.error().is_some());
}

#[test]
fn a_metric_error_stops_evaluation_and_is_reported_as_an_error() {
    let (_, dl_test) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let updates = Arc::new(AtomicUsize::new(0));

    let result = EvaluatorBuilder::new(dir.path())
        .metric(FailingMetric {
            updates: updates.clone(),
            fail_from: 1,
        })
        .with_application_logger(None)
        .build(ToyModel::new(&Device::flex()))
        .eval_all([("first", dl_test.clone()), ("second", dl_test)]);

    assert_eq!(result.interrupted, None, "an error is not an interruption");
    let (_, split) = metric_failure(result.error);
    assert_eq!(split, Split::Test(Some(Arc::new("first".to_string()))));
    // The second dataset is not evaluated once the first one failed.
    assert!(updates.load(Ordering::SeqCst) < 4, "evaluation went on");
}
