//! Integration tests verifying that LrPolicyScheduler correctly freezes parameter groups.

mod common;

use common::*;

use burn_core::{module::ParamGroup, tensor::Device};
use burn_optim::{SgdConfig, lr_scheduler::module_lr_scheduler::ModuleLrSchedulerConfig};
use burn_train::{Learner, SupervisedTraining, logger::InMemoryMetricLogger, metric::LossMetric};

/// Test the integration of [LrPolicyScheduler](burn_optim::lr_scheduler::policy::LrPolicyScheduler) in burn-train.
/// A parameter group with LR=0 must not change after training, while parameters using
/// the default LR must change (given a non-zero gradient).
#[test]
fn frozen_group_param_unchanged_after_training() {
    let device = Device::flex().autodiff();
    let model = TwoLayerModel::new(&device);

    let before_frozen = model
        .frozen
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let before_active = model.active.val().into_data().to_vec::<f32>().unwrap();

    let optim = SgdConfig::new().init();

    let scheduler = ModuleLrSchedulerConfig::new(1e-2.into())
        .with_group(ParamGroup::from_predicate("frozen"), 0.0_f64)
        .init()
        .unwrap();

    let (dl_train, dl_valid) = make_dataloaders();

    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), dl_train, dl_valid)
        .num_epochs(1)
        .with_metric_logger(InMemoryMetricLogger::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_application_logger(None)
        .launch(Learner::new(model, optim, scheduler));

    let after_frozen = result
        .model
        .frozen
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let after_active = result
        .model
        .active
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_eq!(
        before_frozen, after_frozen,
        "frozen param (LR=0) must not change after training"
    );
    assert_ne!(
        before_active, after_active,
        "active param (LR=1e-2) must change after training"
    );
}
