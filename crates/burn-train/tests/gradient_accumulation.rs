mod common;

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use burn_core::{
    data::{
        dataloader::{
            BatchDataLoader, BatchStrategy, DataLoader, DataLoaderBuilder, DataLoaderIterator,
            Progress, TotalCount,
        },
        dataset::{DatasetError, InMemDataset},
    },
    tensor::{Device, Tensor, TensorReadError},
};
use burn_optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};
use burn_train::{
    ExecutionStrategy, Interrupter, Learner, MultiDeviceOptim, SplitOutcome, SupervisedTraining,
    logger::{EvaluationProgressLogger, InMemoryMetricLogger, TrainingProgressLogger},
    metric::{Metric, MetricMetadata, MetricName, SerializedEntry},
};
use common::*;

#[derive(Clone)]
struct CountingScheduler {
    calls: Arc<AtomicUsize>,
}

impl LrScheduler for CountingScheduler {
    fn step(&mut self) -> f64 {
        self.calls.fetch_add(1, Ordering::SeqCst);
        0.1
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(&mut self, _record: LrSchedulerRecord) {}
}

#[test]
fn partial_accumulation_window_updates_model_once() {
    let device = Device::flex().autodiff();
    let model = ToyModel {
        weight: burn_core::module::Param::from_tensor(Tensor::from_data([[1.0, 2.0]], &device)),
    };
    let before = model.weight.val().try_into_vec_as::<f32>().unwrap();

    let optim = burn_optim::SgdConfig::new().init();
    let calls = Arc::new(AtomicUsize::new(0));
    let scheduler = CountingScheduler {
        calls: calls.clone(),
    };
    let learner = Learner::new(model, optim, scheduler);

    // Two training batches form one partial window when the interval is three.
    let (dl_train, dl_valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();

    let result = SupervisedTraining::new(dir.path(), dl_train, dl_valid)
        .num_epochs(1)
        .grads_accumulation(3)
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);

    let after = result.model.weight.val().try_into_vec_as::<f32>().unwrap();

    assert!(result.error.is_none());
    assert_eq!(before, [1.0, 2.0]);
    for (actual, expected) in after.iter().zip([0.8, 1.6]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the scheduler must advance once per optimizer update"
    );
}

#[test]
#[should_panic(expected = "Gradient accumulation must be greater than zero.")]
fn zero_accumulation_interval_is_rejected() {
    let (dl_train, dl_valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();

    let _ =
        SupervisedTraining::<ToyModel>::new(dir.path(), dl_train, dl_valid).grads_accumulation(0);
}

#[rstest::rstest]
#[case::single(None, [2.42, 2.996])]
#[case::main_device(Some(MultiDeviceOptim::OptimMainDevice), [2.42, 2.96])]
#[case::sharded(Some(MultiDeviceOptim::OptimSharded), [2.42, 2.96])]
fn accumulation_sums_distinct_gradients_in_full_and_partial_windows(
    #[case] mode: Option<MultiDeviceOptim>,
    #[case] expected: [f32; 2],
) {
    struct TargetBatcher;
    impl burn_core::data::dataloader::batcher::Batcher<f32, DummyBatch> for TargetBatcher {
        fn batch(&self, items: Vec<f32>, device: &Device) -> DummyBatch {
            DummyBatch {
                target: Tensor::from_data([[items[0]; 2]], device),
            }
        }
    }

    let device = Device::flex().autodiff();
    let model = ToyModel {
        weight: burn_core::module::Param::from_tensor(Tensor::from_data([[1.0, 2.0]], &device)),
    };
    let learner = Learner::new(model, burn_optim::SgdConfig::new().init(), 0.1);
    let train = DataLoaderBuilder::new(TargetBatcher)
        .batch_size(1)
        .build(InMemDataset::new(vec![0.0, 2.0, 4.0, 6.0, 8.0]));
    let (_, valid) = make_dataloaders();
    let strategy = match mode {
        None => ExecutionStrategy::single(device),
        Some(mode) => ExecutionStrategy::multi(vec![device.clone(), device], mode),
    };
    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .num_epochs(1)
        .grads_accumulation(2)
        .with_training_strategy(strategy.into())
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);
    assert!(result.error.is_none());
    // Each batch has gradient w - target. Single-device windows are [0,2], [4,6], [8].
    // Two devices process [0,2] and [4,6,8], so their windows are [0,4,2,6], [8].
    for (actual, expected) in result
        .model
        .weight
        .val()
        .try_into_vec_as::<f32>()
        .unwrap()
        .iter()
        .zip(expected)
    {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn multi_device_progress_retains_exhausted_loaders() {
    let device = Device::flex().autodiff();
    let model = ToyModel::new(&device);
    let loaders: Vec<_> = [4, 6]
        .into_iter()
        .map(|len| {
            DataLoaderBuilder::new(DummyBatcher)
                .batch_size(2)
                .set_device(device.clone())
                .build(InMemDataset::new(vec![(); len]))
        })
        .collect();
    let mut iterators: Vec<_> = loaders.iter().map(|loader| loader.iter()).collect();
    let step = burn_train::train::MultiDevicesTrainStep::<ToyModel>::new(&[device.clone(), device]);

    for expected in [4, 8, 10] {
        let (items, progress) = step.step(&mut iterators, &model).unwrap();
        assert!(!items.is_empty());
        assert_eq!(progress.items_processed, expected);
        assert_eq!(progress.total, TotalCount::Exact(10));
    }
    let (items, progress) = step.step(&mut iterators, &model).unwrap();
    assert!(items.is_empty());
    assert_eq!(progress.items_processed, 10);
    assert_eq!(progress.total, TotalCount::Exact(10));
}

// Buffer the entire dataset before yielding pairs, optionally dropping an incomplete tail.
struct BufferedPairs {
    items: Vec<()>,
    drop_last: bool,
}

impl BatchStrategy<()> for BufferedPairs {
    fn add(&mut self, item: ()) {
        self.items.push(item);
    }

    fn batch(&mut self, force: bool) -> Option<Vec<()>> {
        if !force || self.items.is_empty() {
            return None;
        }
        if self.drop_last && self.items.len() < 2 {
            self.items.clear();
            return None;
        }
        Some(self.items.drain(..self.items.len().min(2)).collect())
    }

    fn clone_dyn(&self) -> Box<dyn BatchStrategy<()>> {
        Box::new(Self {
            items: Vec::new(),
            drop_last: self.drop_last,
        })
    }

    fn batch_size(&self) -> Option<usize> {
        Some(2)
    }
}

#[derive(Clone, Copy)]
enum EndCondition {
    Exhausted,
    Error,
    Interrupted,
}

// Exercise custom loader contracts independently of the built-in emitted-item counter.
struct CustomLoader {
    inner: Arc<dyn DataLoader<DummyBatch>>,
    reports_full: bool,
    end: EndCondition,
    interrupter: Interrupter,
}

impl DataLoader<DummyBatch> for CustomLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<DummyBatch> + 'a> {
        Box::new(CustomIterator {
            inner: self.inner.iter(),
            loader: self,
            batches: 0,
        })
    }

    fn num_items(&self) -> usize {
        self.inner.num_items()
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<DummyBatch>> {
        Arc::new(Self {
            inner: self.inner.to_device(device),
            reports_full: self.reports_full,
            end: self.end,
            interrupter: self.interrupter.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<DummyBatch>> {
        Arc::new(Self {
            inner: self.inner.slice(start, end),
            reports_full: self.reports_full,
            end: self.end,
            interrupter: self.interrupter.clone(),
        })
    }
}

struct CustomIterator<'a> {
    inner: Box<dyn DataLoaderIterator<DummyBatch> + 'a>,
    loader: &'a CustomLoader,
    batches: usize,
}

impl Iterator for CustomIterator<'_> {
    type Item = Result<DummyBatch, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batches == 1 {
            match self.loader.end {
                EndCondition::Exhausted => {}
                EndCondition::Error => {
                    return Some(Err(DatasetError::new(std::io::Error::other("test error"))));
                }
                EndCondition::Interrupted => {
                    self.loader.interrupter.stop(Some("test interruption"));
                    return None;
                }
            }
        }
        let item = self.inner.next();
        if item.is_some() {
            self.batches += 1;
        }
        item
    }
}

impl DataLoaderIterator<DummyBatch> for CustomIterator<'_> {
    fn progress(&self) -> Progress {
        let mut progress = self.inner.progress();
        if self.loader.reports_full {
            progress.total = TotalCount::Exact(self.loader.num_items());
            progress.items_processed = self.loader.num_items();
        }
        progress
    }
}

#[rstest::rstest]
#[case::misleading_progress(10, false, true, 3, 2, 1)]
#[case::dropped_tail(23, true, false, 4, 3, 2)]
#[case::full_window(12, false, false, 3, 2, 1)]
#[case::empty(0, false, false, 3, 0, 0)]
fn accumulation_uses_exhaustion_not_progress(
    #[values(
        None,
        Some(MultiDeviceOptim::OptimMainDevice),
        Some(MultiDeviceOptim::OptimSharded)
    )]
    mode: Option<MultiDeviceOptim>,
    #[case] len: usize,
    #[case] drop_last: bool,
    #[case] reports_full: bool,
    #[case] accumulation: usize,
    #[case] single_updates: usize,
    #[case] multi_updates: usize,
) {
    check_accumulation(
        mode,
        len,
        drop_last,
        reports_full,
        accumulation,
        EndCondition::Exhausted,
        if mode.is_some() {
            multi_updates
        } else {
            single_updates
        },
    );
}

#[rstest::rstest]
#[case::error(EndCondition::Error)]
#[case::interrupted(EndCondition::Interrupted)]
fn aborted_epoch_does_not_flush_partial_window(
    #[values(
        None,
        Some(MultiDeviceOptim::OptimMainDevice),
        Some(MultiDeviceOptim::OptimSharded)
    )]
    mode: Option<MultiDeviceOptim>,
    #[case] end: EndCondition,
) {
    check_accumulation(mode, 12, false, true, 3, end, 0);
}

fn check_accumulation(
    mode: Option<MultiDeviceOptim>,
    len: usize,
    drop_last: bool,
    reports_full: bool,
    accumulation: usize,
    end: EndCondition,
    expected_updates: usize,
) {
    let device = Device::flex().autodiff();
    let model = ToyModel::new(&device);
    let before = model.weight.val().try_into_vec_as::<f32>().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let learner = Learner::new(
        model,
        burn_optim::SgdConfig::new().init(),
        CountingScheduler {
            calls: calls.clone(),
        },
    );
    let interrupter = Interrupter::new();
    let train = Arc::new(CustomLoader {
        inner: Arc::new(BatchDataLoader::new(
            Box::new(BufferedPairs {
                items: Vec::new(),
                drop_last,
            }),
            Arc::new(InMemDataset::new(vec![(); len])),
            Arc::new(DummyBatcher),
            device.clone(),
            None,
        )),
        reports_full,
        end,
        interrupter: interrupter.clone(),
    });
    let (_, valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let strategy = match mode {
        None => ExecutionStrategy::single(device),
        Some(mode) => ExecutionStrategy::multi(vec![device.clone(), device], mode),
    };
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .num_epochs(1)
        .grads_accumulation(accumulation)
        .with_interrupter(interrupter)
        .with_progress_logger(report)
        .with_training_strategy(strategy.into())
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);
    let events = events.lock().unwrap();
    let train_updates = events
        .iter()
        .filter(|event| matches!(event, Report::Update(split) if split == "train"))
        .count();
    let iterations = events
        .iter()
        .filter(|event| matches!(event, Report::Iteration(split) if split == "train"))
        .count();
    assert_eq!(
        train_updates, iterations,
        "ending a split must not create an iteration"
    );
    let endings: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Report::End(name, progress, outcome) if name == "train" => Some((progress, outcome)),
            _ => None,
        })
        .collect();
    assert_eq!(endings.len(), 1);
    let (final_progress, outcome) = endings[0];
    assert_eq!(
        *outcome,
        match end {
            EndCondition::Exhausted => SplitOutcome::Completed,
            EndCondition::Error => SplitOutcome::Failed,
            EndCondition::Interrupted => SplitOutcome::Interrupted,
        }
    );
    if matches!(end, EndCondition::Exhausted) && !reports_full {
        assert_eq!(
            final_progress.items_processed,
            if drop_last { len / 2 * 2 } else { len }
        );
        assert_eq!(
            final_progress.total,
            TotalCount::Exact(final_progress.items_processed)
        );
    }
    let after = result.model.weight.val().try_into_vec_as::<f32>().unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        expected_updates,
        "len={len}, drop_last={drop_last}, reports_full={reports_full}, mode={mode:?}"
    );
    if expected_updates == 0 {
        assert_eq!(
            before, after,
            "empty and aborted windows must not update the model"
        );
    } else {
        assert_ne!(before, after, "completed windows must update the model");
    }
}

#[derive(Debug)]
enum Report {
    Start,
    Update(String),
    Iteration(String),
    Epoch,
    End(String, Progress, SplitOutcome),
}

#[derive(Default)]
struct RecordingProgress {
    split: String,
    events: Arc<Mutex<Vec<Report>>>,
}

impl TrainingProgressLogger for RecordingProgress {
    fn start(&mut self, _: usize, _: usize, _: Option<usize>, _: Option<&str>) {}
    fn update_epoch(&mut self, _: usize) {
        self.events.lock().unwrap().push(Report::Epoch);
    }
    fn start_split(&mut self, split: &str, _: &Progress) {
        self.split = split.into();
        self.events.lock().unwrap().push(Report::Start);
    }
    fn update_split(&mut self, _: &Progress) {
        self.events
            .lock()
            .unwrap()
            .push(Report::Update(self.split.clone()));
    }
    fn end_split(&mut self, progress: &Progress, outcome: SplitOutcome) {
        self.events.lock().unwrap().push(Report::End(
            self.split.clone(),
            progress.clone(),
            outcome,
        ));
    }
    fn end(&mut self) {}
    fn log_event_training(&mut self, event: String) {
        if event == "Iteration" {
            self.events
                .lock()
                .unwrap()
                .push(Report::Iteration(self.split.clone()));
        }
    }
}

impl EvaluationProgressLogger for RecordingProgress {
    fn start_global_progress(&mut self, _: usize) {}
    fn start_test(&mut self, name: &str, progress: &Progress) {
        self.start_split(name, progress);
    }
    fn update_test_progress(&mut self, progress: &Progress) {
        self.update_split(progress);
    }
    fn end_test(&mut self, progress: &Progress, outcome: SplitOutcome) {
        self.end_split(progress, outcome);
    }
    fn end_global_progress(&mut self) {}
    fn log_event_evaluation(&mut self, event: String) {
        self.log_event_training(event);
    }
}

#[rstest::rstest]
#[case::empty(0)]
#[case::dropped_tail(5)]
fn evaluation_reports_completion_without_an_extra_iteration(#[case] len: usize) {
    let device = Device::flex();
    let model = ToyModel::new(&device);
    let loader = Arc::new(BatchDataLoader::new(
        Box::new(BufferedPairs {
            items: vec![],
            drop_last: true,
        }),
        Arc::new(InMemDataset::new(vec![(); len])),
        Arc::new(DummyBatcher),
        device,
        None,
    ));
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let dir = tempfile::tempdir().unwrap();
    burn_train::EvaluatorBuilder::new(dir.path())
        .with_progress_logger(report)
        .with_application_logger(None)
        .build(model)
        .eval("test", loader);
    let events = events.lock().unwrap();
    let updates = events
        .iter()
        .filter(|event| matches!(event, Report::Update(_)))
        .count();
    let endings: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Report::End(_, progress, outcome) => Some((progress, outcome)),
            _ => None,
        })
        .collect();
    assert_eq!(endings.len(), 1);
    let (final_progress, outcome) = endings[0];
    assert_eq!(*outcome, SplitOutcome::Completed);
    assert_eq!(final_progress.items_processed, len / 2 * 2);
    assert_eq!(final_progress.total, TotalCount::Exact(len / 2 * 2));
    assert_eq!(updates, len / 2);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, Report::Iteration(_)))
            .count(),
        len / 2
    );
    assert!(matches!(events.last(), Some(Report::End(_, _, _))));
}

// Fail either while processing the last batch or while finalizing its metrics. Neither
// path may announce successful completion, even though the loader reached its exact total.
#[derive(Clone)]
struct FailsAtEnd {
    compute: bool,
}

impl Metric for FailsAtEnd {
    type Input = ();

    fn name(&self) -> MetricName {
        Arc::new("FailsAtEnd".to_string())
    }

    fn update(
        &mut self,
        _: &(),
        metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        if !self.compute && metadata.iteration == Some(2) {
            return Err(burn_std::ExecutionError::with_context("last batch failed").into());
        }
        Ok(SerializedEntry::new("ok".into(), "ok".into()))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        if self.compute {
            return Err(burn_std::ExecutionError::with_context("final metric failed").into());
        }
        Ok(SerializedEntry::new("ok".into(), "ok".into()))
    }

    fn clear(&mut self) {}
}

#[rstest::rstest]
#[case(false, false)]
#[case(true, false)]
#[case(true, true)]
fn metric_failures_reach_the_terminal_callback(#[case] compute: bool, #[case] valid_failure: bool) {
    let (train, valid) = make_dataloaders();
    let learner = make_learner(&Device::flex().autodiff());
    let before = learner
        .model()
        .weight
        .val()
        .try_into_vec_as::<f32>()
        .unwrap();
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let dir = tempfile::tempdir().unwrap();
    let training = SupervisedTraining::new(dir.path(), train, valid);
    let training = if valid_failure {
        training.metric_valid(FailsAtEnd { compute })
    } else {
        training.metric_train(FailsAtEnd { compute })
    };
    let result = training
        .num_epochs(2)
        .grads_accumulation(3)
        .with_progress_logger(report)
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);
    assert!(result.error.is_some());
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Report::Start))
            .count(),
        if valid_failure { 2 } else { 1 },
        "a failed split must not start another split or epoch"
    );
    assert!(!events.iter().any(|event| matches!(event, Report::Epoch)));
    assert!(
        matches!(events.last(), Some(Report::End(_, progress, SplitOutcome::Failed))
        if progress.items_processed == 4 && progress.total == TotalCount::Exact(4))
    );
    if !compute {
        assert_eq!(
            before,
            result.model.weight.val().try_into_vec_as::<f32>().unwrap(),
            "a pending metric failure must prevent the final partial optimizer update"
        );
    }
}

#[test]
fn evaluation_failure_at_the_exact_total_is_not_completion() {
    let (_, loader) = make_dataloaders();
    let report = RecordingProgress::default();
    let events = report.events.clone();
    let dir = tempfile::tempdir().unwrap();
    let result = burn_train::EvaluatorBuilder::new(dir.path())
        .metric(FailsAtEnd { compute: false })
        .with_progress_logger(report)
        .with_application_logger(None)
        .build(ToyModel::new(&Device::flex()))
        .eval_all([("first", loader.clone()), ("second", loader)]);
    assert!(result.error.is_some());
    let events = events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Report::Start))
            .count(),
        1
    );
    assert!(
        matches!(events.last(), Some(Report::End(_, progress, SplitOutcome::Failed))
        if progress.items_processed == 4 && progress.total == TotalCount::Exact(4))
    );
}
