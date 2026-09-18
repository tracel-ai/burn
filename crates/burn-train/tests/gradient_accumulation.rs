mod common;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use burn_core::tensor::Device;
use burn_optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};
use burn_train::{Learner, SupervisedTraining, logger::InMemoryMetricLogger};
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
    let model = ToyModel::new(&device);
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

    assert_ne!(before, after, "the partial window must update the model");
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
