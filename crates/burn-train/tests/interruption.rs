mod common;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use burn_core::{
    data::dataloader::Progress,
    tensor::{Device, TensorReadError},
};
use burn_std::ExecutionError;
use burn_train::{
    EvaluatorBuilder, EventProcessorFailure, Interrupter, Interruption, SupervisedTraining,
    TrainingError,
    logger::{EvaluationProgressLogger, InMemoryMetricLogger, TrainingProgressLogger},
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

    let report = RecordingProgress::default();
    let events = report.events.clone();
    let result = EvaluatorBuilder::new(dir.path())
        .metric(FailingMetric {
            updates: updates.clone(),
            fail_from: 2,
        })
        .with_progress_logger(report)
        .with_application_logger(None)
        .build(ToyModel::new(&Device::flex()))
        .eval_all([("first", dl_test.clone()), ("second", dl_test)]);

    assert_eq!(result.interrupted, None, "an error is not an interruption");
    let (_, split) = metric_failure(result.error);
    assert_eq!(split, Split::Test(Some(Arc::new("first".to_string()))));
    // The second dataset is not evaluated once the first one failed.
    assert!(updates.load(Ordering::SeqCst) < 4, "evaluation went on");
    let events = events.lock().unwrap();
    assert_eq!(events.starts, ["first"]);
    assert!(matches!(events.ends.as_slice(), [progress]
        if progress.items_processed == 4 && progress.total == Some(4)));
}

#[derive(Default)]
struct Reports {
    starts: Vec<String>,
    ends: Vec<Progress>,
    updates: usize,
    iterations: usize,
    epochs: usize,
}

#[derive(Default)]
struct RecordingProgress {
    events: Arc<Mutex<Reports>>,
}

impl TrainingProgressLogger for RecordingProgress {
    fn start(&mut self, _: usize, _: usize, _: Option<usize>, _: Option<&str>) {}
    fn start_split(&mut self, split: &str, _: &Progress) {
        self.events.lock().unwrap().starts.push(split.into());
    }
    fn update_split(&mut self, _: &Progress) {
        self.events.lock().unwrap().updates += 1;
    }
    fn end_split(&mut self, progress: &Progress) {
        self.events.lock().unwrap().ends.push(progress.clone());
    }
    fn update_epoch(&mut self, _: usize) {
        self.events.lock().unwrap().epochs += 1;
    }
    fn log_event_training(&mut self, event: String) {
        if event == "Iteration" {
            self.events.lock().unwrap().iterations += 1;
        }
    }
    fn end(&mut self) {}
}

impl EvaluationProgressLogger for RecordingProgress {
    fn start_global_progress(&mut self, _: usize) {}
    fn start_test(&mut self, name: &str, progress: &Progress) {
        self.start_split(name, progress);
    }
    fn update_test_progress(&mut self, progress: &Progress) {
        self.update_split(progress);
    }
    fn end_test(&mut self, progress: &Progress) {
        self.end_split(progress);
    }
    fn log_event_evaluation(&mut self, event: String) {
        self.log_event_training(event);
    }
    fn end_global_progress(&mut self) {}
}

#[rstest::rstest]
#[case::completed(usize::MAX)]
#[case::last_batch(2)]
fn training_stops_on_metric_errors(#[case] fail_from: usize) {
    let learner = make_learner(&Device::flex().autodiff());
    let before = learner
        .model()
        .weight
        .val()
        .try_into_vec_as::<f32>()
        .unwrap();
    let (train, validation) = make_dataloaders();
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let dir = tempfile::tempdir().unwrap();
    let metric = FailingMetric {
        updates: Arc::new(AtomicUsize::new(0)),
        fail_from,
    };
    let result = SupervisedTraining::new(dir.path(), train, validation)
        .metric_train(metric)
        .num_epochs(2)
        .grads_accumulation(3)
        .with_progress_logger(report)
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);
    assert_eq!(result.interrupted, None, "an error is not an interruption");
    if fail_from == usize::MAX {
        assert!(result.error.is_none());
        assert_eq!(events.lock().unwrap().epochs, 2);
        return;
    }
    let (metric, split) = metric_failure(result.error);
    assert_eq!(metric, "Failing");
    assert_eq!(split, Split::Train);
    let events = events.lock().unwrap();
    assert_eq!(events.starts, ["train"]);
    assert_eq!(events.ends.len(), events.starts.len());
    assert_eq!(events.epochs, 0);
    assert!(matches!(events.ends.last(), Some(progress)
        if progress.items_processed == 4 && progress.total == Some(4)));
    assert_eq!(
        before,
        result.model.weight.val().try_into_vec_as::<f32>().unwrap(),
        "the final batch's metric error must prevent the partial optimizer update"
    );
}

#[rstest::rstest]
#[case::training(true)]
#[case::evaluation(false)]
fn empty_splits_finish_without_iterations(#[case] training: bool) {
    let (train, valid) = make_dataloaders();
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let dir = tempfile::tempdir().unwrap();
    let error = if training {
        SupervisedTraining::new(dir.path(), train.slice(0, 0), valid.slice(0, 0))
            .num_epochs(1)
            .with_progress_logger(report)
            .with_metric_logger(InMemoryMetricLogger::new())
            .with_application_logger(None)
            .launch(make_learner(&Device::flex().autodiff()))
            .error
    } else {
        EvaluatorBuilder::new(dir.path())
            .with_progress_logger(report)
            .with_application_logger(None)
            .build(ToyModel::new(&Device::flex()))
            .eval("empty", valid.slice(0, 0))
            .error
    };
    assert!(error.is_none());
    let events = events.lock().unwrap();
    assert_eq!(events.ends.len(), if training { 2 } else { 1 });
    assert_eq!(events.updates, 0);
    assert_eq!(events.iterations, 0);
    for progress in &events.ends {
        assert_eq!(progress.items_processed, 0);
        assert_eq!(progress.total, Some(0));
    }
}
