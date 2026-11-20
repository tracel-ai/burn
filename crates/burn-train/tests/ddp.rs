//! Flex has no collectives, so these run only with a `test-*` backend feature, for example
//! `cargo test -p burn-train --features test-metal --test ddp`.
#![cfg(any(feature = "test-cuda", feature = "test-metal", feature = "test-vulkan"))]

use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::Duration;

use burn_core as burn;

use burn_core::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::{Module, Param},
    tensor::{
        Device, Distribution, Tensor,
        distributed::{DistributedConfig, ReduceOperation},
    },
};
use burn_optim::{ModuleOptimizer, SgdConfig, lr_scheduler::constant::ConstantLr};
use burn_train::{
    ExecutionStrategy, InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainOutput,
    TrainStep, TrainingError, TrainingStrategy, logger::InMemoryMetricLogger,
};

#[derive(Module, Debug)]
struct PanickingModel {
    weight: Param<Tensor<2>>,
}

#[derive(Clone, Debug)]
struct Batch {
    target: Tensor<2>,
    fail: bool,
}

/// Marks the batch holding `failing_item` so the training step panics on it.
struct FlagBatcher {
    failing_item: Option<usize>,
}

impl Batcher<usize, Batch> for FlagBatcher {
    fn batch(&self, items: Vec<usize>, device: &Device) -> Batch {
        Batch {
            target: Tensor::zeros([1, 2], device),
            fail: self.failing_item.is_some_and(|item| items.contains(&item)),
        }
    }
}

impl PanickingModel {
    fn loss(&self, item: Batch) -> RegressionOutput {
        let output = self.weight.val();
        let loss = output
            .clone()
            .sub(item.target.clone())
            .powi_scalar(2)
            .mean();
        RegressionOutput::new(loss, output, item.target)
    }
}

impl TrainStep for PanickingModel {
    type Input = Batch;
    type Output = RegressionOutput;

    fn step(&self, item: Batch) -> TrainOutput<RegressionOutput> {
        if item.fail {
            panic!("injected training step failure");
        }
        let output = self.loss(item);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl InferenceStep for PanickingModel {
    type Input = Batch;
    type Output = RegressionOutput;

    fn step(&self, item: Batch) -> RegressionOutput {
        self.loss(item)
    }
}

fn dataloader(failing_item: Option<usize>) -> Arc<dyn DataLoader<Batch>> {
    DataLoaderBuilder::new(FlagBatcher { failing_item })
        .batch_size(1)
        .build(InMemDataset::new((0..4).collect()))
}

/// Trains on two DDP workers, which split the four items in order: the main worker gets items
/// 0 and 1, the secondary worker 2 and 3. The healthy worker reaches the collective of the
/// failing step and waits for gradients that never come.
fn train_failing_on(failing_item: usize) -> Option<Arc<TrainingError>> {
    // One distributed context may run per backend, so the cases cannot overlap.
    static CONTEXT: Mutex<()> = Mutex::new(());
    let _context = CONTEXT.lock().unwrap_or_else(PoisonError::into_inner);
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        // Prefers the enabled CubeCL runtime over Flex.
        let device = Device::default();
        let model = PanickingModel {
            weight: Param::from_tensor(Tensor::random(
                [1, 2],
                Distribution::Default,
                &device.clone().autodiff(),
            )),
        };
        let optim: ModuleOptimizer = SgdConfig::new().init();
        let learner = Learner::new(model, optim, ConstantLr::new(1e-3));
        let strategy = ExecutionStrategy::ddp(
            vec![device.clone(), device],
            DistributedConfig {
                all_reduce_op: ReduceOperation::Mean,
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let result =
            SupervisedTraining::new(dir.path(), dataloader(Some(failing_item)), dataloader(None))
                .num_epochs(2)
                .with_training_strategy(TrainingStrategy::Default(strategy))
                .with_metric_logger(InMemoryMetricLogger::new())
                .with_application_logger(None)
                .launch(learner);
        sender.send(result.error).unwrap();
    });
    receiver
        .recv_timeout(Duration::from_secs(60))
        .expect("DDP training must return when a worker fails instead of hanging")
}

#[rstest::rstest]
#[case::main_worker(1, 0)]
#[case::secondary_worker(3, 1)]
fn a_failing_worker_ends_ddp_training_with_an_error(
    #[case] failing_item: usize,
    #[case] failing_worker: usize,
) {
    match train_failing_on(failing_item).as_deref() {
        Some(TrainingError::Workers(failures)) => {
            let [failure] = failures.as_slice() else {
                panic!("expected only the root failure, got {failures:?}");
            };
            assert_eq!(failure.device_id, failing_worker);
            assert!(failure.message.contains("injected training step failure"));
        }
        other => panic!("expected a worker failure, got {other:?}"),
    }
}
