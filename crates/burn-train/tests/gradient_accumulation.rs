mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use burn_core::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, DataLoaderIterator, Progress},
        dataset::{DatasetError, InMemDataset},
    },
    tensor::{Device, Tensor},
};
use burn_optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};
use burn_train::{
    ExecutionStrategy, Interrupter, Learner, MultiDeviceOptim, SupervisedTraining,
    logger::InMemoryMetricLogger,
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
fn accumulation_respects_exhaustion_and_abort(
    #[case] mode: Option<MultiDeviceOptim>,
    #[case] expected: [f32; 2],
    #[values(
        EndCondition::Exhausted,
        EndCondition::Error,
        EndCondition::Interrupted
    )]
    end: EndCondition,
) {
    let device = Device::flex().autodiff();
    let model = ToyModel {
        weight: burn_core::module::Param::from_tensor(Tensor::from_data([[1.0, 2.0]], &device)),
    };
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
        targets: vec![0.0, 2.0, 4.0, 6.0, 8.0],
        device: device.clone(),
        position: 0,
        end,
        interrupter: interrupter.clone(),
    });
    let (_, valid) = make_dataloaders();
    let strategy = match mode {
        None => ExecutionStrategy::single(device),
        Some(mode) => ExecutionStrategy::multi(vec![device.clone(), device], mode),
    };
    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .num_epochs(1)
        .grads_accumulation(2)
        .with_interrupter(interrupter)
        .with_training_strategy(strategy.into())
        .with_metric_logger(InMemoryMetricLogger::new())
        .with_application_logger(None)
        .launch(learner);
    assert_eq!(result.error.is_some(), matches!(end, EndCondition::Error));
    assert_eq!(
        result.interrupted.is_some(),
        matches!(end, EndCondition::Interrupted)
    );
    let (expected, updates) = match end {
        EndCondition::Exhausted => (expected, if mode.is_none() { 3 } else { 2 }),
        EndCondition::Error | EndCondition::Interrupted => ([1.0, 2.0], 0),
    };
    assert_eq!(calls.load(Ordering::SeqCst), updates);
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
        assert_eq!(progress.total, Some(10));
    }
    let (items, progress) = step.step(&mut iterators, &model).unwrap();
    assert!(items.is_empty());
    assert_eq!(progress.items_processed, 10);
    assert_eq!(progress.total, Some(10));
}

#[derive(Clone, Copy)]
enum EndCondition {
    Exhausted,
    Error,
    Interrupted,
}

// A synthetic loader whose progress claims completion before its batches are exhausted.
#[derive(Clone)]
struct CustomLoader {
    targets: Vec<f32>,
    device: Device,
    position: usize,
    end: EndCondition,
    interrupter: Interrupter,
}

impl DataLoader<DummyBatch> for CustomLoader {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<DummyBatch> + 'a> {
        Box::new(self.clone())
    }

    fn num_items(&self) -> usize {
        self.targets.len()
    }

    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<DummyBatch>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<DummyBatch>> {
        Arc::new(Self {
            targets: self.targets[start..end].to_vec(),
            ..self.clone()
        })
    }
}

impl Iterator for CustomLoader {
    type Item = Result<DummyBatch, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position == 1 {
            match self.end {
                EndCondition::Exhausted => {}
                EndCondition::Error => {
                    return Some(Err(DatasetError::new(std::io::Error::other("test error"))));
                }
                EndCondition::Interrupted => {
                    self.interrupter.stop(Some("test interruption"));
                    return None;
                }
            }
        }
        let target = *self.targets.get(self.position)?;
        self.position += 1;
        Some(Ok(DummyBatch {
            target: Tensor::from_data([[target; 2]], &self.device),
        }))
    }
}

impl DataLoaderIterator<DummyBatch> for CustomLoader {
    fn progress(&self) -> Progress {
        Progress::new(self.targets.len(), self.targets.len(), None)
    }
}
