//! Integration tests for checkpoint saving/loading in a minimal supervised training loop.

mod common;
use std::fs;

use common::*;

use burn_core::tensor::Device;
use burn_train::{
    SupervisedTraining, checkpoint::KeepLastNCheckpoints, logger::InMemoryMetricLogger,
    metric::LossMetric,
};

#[test]
fn checkpoint_saves_bpk_files() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = dir.path().to_path_buf();
    let checkpoint_dir = dir_path.join("checkpoint");

    let device = Device::flex().autodiff();
    let (dl_train, dl_valid) = make_dataloaders();

    SupervisedTraining::new(&dir_path, dl_train, dl_valid)
        .num_epochs(3)
        .with_default_checkpointers()
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(3))
        .with_metric_logger(InMemoryMetricLogger::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_application_logger(None)
        .launch(make_learner(&device));

    for epoch in 1..=3 {
        let model_path = checkpoint_dir.join(format!("model-{epoch}.bpk"));
        assert!(
            model_path.exists(),
            "expected checkpoint file: {}",
            model_path.display()
        );
    }
}

/// Training can resume from a saved checkpoint: `SupervisedTraining::checkpoint(epoch)` loads
/// the model/optimizer/scheduler and continues from the next epoch.
#[test]
fn checkpoint_resume_continues_training() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = dir.path().to_path_buf();

    let device = Device::flex().autodiff();

    {
        let (dl_train, dl_valid) = make_dataloaders();
        SupervisedTraining::new(&dir_path, dl_train, dl_valid)
            .num_epochs(2)
            .with_default_checkpointers()
            .with_checkpointing_strategy(KeepLastNCheckpoints::new(2))
            .with_metric_logger(InMemoryMetricLogger::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_application_logger(None)
            .launch(make_learner(&device));
    }

    let ckpt = dir_path.join("checkpoint").join("model-2.bpk");
    assert!(ckpt.exists(), "epoch-2 checkpoint must exist before resume");

    // Resume from epoch 2 and train up to epoch 4.
    {
        let (dl_train, dl_valid) = make_dataloaders();
        SupervisedTraining::new(&dir_path, dl_train, dl_valid)
            .num_epochs(4)
            .checkpoint(2)
            .with_default_checkpointers()
            .with_checkpointing_strategy(KeepLastNCheckpoints::new(4))
            .with_metric_logger(InMemoryMetricLogger::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_application_logger(None)
            .launch(make_learner(&device));
    }

    for epoch in 1..=4 {
        let path = dir_path
            .join("checkpoint")
            .join(format!("model-{epoch}.bpk"));
        assert!(
            path.exists(),
            "expected checkpoint file for epoch {epoch}: {}",
            path.display()
        );
    }
}

#[test]
fn checkpoint_restores_model_weights() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = dir.path().to_path_buf();

    let device = Device::flex().autodiff();
    let (dl_train, dl_valid) = make_dataloaders();

    let result = SupervisedTraining::new(&dir_path, dl_train, dl_valid)
        .num_epochs(1)
        .with_default_checkpointers()
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(1))
        .with_metric_logger(InMemoryMetricLogger::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_application_logger(None)
        .launch(make_learner(&device));

    let trained_weights = result
        .model
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // Load the checkpoint into a fresh learner
    use burn_train::{
        LearningCheckpointer,
        checkpoint::{AsyncCheckpointer, FileCheckpointer},
    };

    let ckpt_dir = dir_path.join("checkpoint");
    let checkpointer = LearningCheckpointer::new(
        AsyncCheckpointer::new(FileCheckpointer::new(&ckpt_dir, "model")),
        AsyncCheckpointer::new(FileCheckpointer::new(&ckpt_dir, "optim")),
        AsyncCheckpointer::new(FileCheckpointer::new(&ckpt_dir, "scheduler")),
        Box::new(KeepLastNCheckpoints::new(1)),
    );

    let fresh = make_learner(&device);
    let restored = checkpointer.load_checkpoint(fresh, 1);

    let restored_weights = restored
        .model()
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_eq!(
        trained_weights, restored_weights,
        "restored weights must match the saved checkpoint"
    );
}

#[test]
fn file_metric_logger_creates_log_directories() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = dir.path().to_path_buf();

    let device = Device::flex().autodiff();
    let (dl_train, dl_valid) = make_dataloaders();

    use burn_train::logger::FileMetricLogger;

    SupervisedTraining::new(&dir_path, dl_train, dl_valid)
        .num_epochs(2)
        .with_metric_logger(FileMetricLogger::new(&dir_path))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_application_logger(None)
        .launch(make_learner(&device));

    for epoch in 1..=2 {
        let train_dir = dir_path.join("train").join(format!("epoch-{epoch}"));
        let valid_dir = dir_path.join("valid").join(format!("epoch-{epoch}"));

        assert!(
            train_dir.exists(),
            "missing train log dir for epoch {epoch}: {}",
            train_dir.display()
        );
        assert!(
            valid_dir.exists(),
            "missing valid log dir for epoch {epoch}: {}",
            valid_dir.display()
        );
    }
}

#[test]
fn file_metric_logger_resumes_logging_at_checkpoint() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let dir_path = dir.path().to_path_buf();

    let device = Device::flex().autodiff();
    let (dl_train, dl_valid) = make_dataloaders();

    use burn_train::logger::FileMetricLogger;

    {
        SupervisedTraining::new(&dir_path, dl_train, dl_valid)
            .num_epochs(1)
            .with_metric_logger(FileMetricLogger::new(&dir_path))
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_default_checkpointers()
            .with_application_logger(None)
            .launch(make_learner(&device));
    }

    let loss_file_train = dir_path.join("train").join("epoch-1/Loss.log");
    let loss_file_valid = dir_path.join("valid").join("epoch-1/Loss.log");

    let content_train_expected = fs::read(loss_file_train).expect("Cannot read file.");
    let content_valid_expected = fs::read(loss_file_valid).expect("Cannot read file.");

    // Resume from epoch 1 and train up to epoch 2.
    {
        let (dl_train, dl_valid) = make_dataloaders();
        SupervisedTraining::new(&dir_path, dl_train, dl_valid)
            .num_epochs(2)
            .checkpoint(1)
            .with_default_checkpointers()
            .with_metric_logger(FileMetricLogger::new(&dir_path))
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_application_logger(None)
            .launch(make_learner(&device));
    }

    for epoch in 1..=2 {
        let train_dir = dir_path.join("train").join(format!("epoch-{epoch}"));
        let valid_dir = dir_path.join("valid").join(format!("epoch-{epoch}"));

        assert!(
            train_dir.exists(),
            "missing train log dir for epoch {epoch}: {}",
            train_dir.display()
        );
        assert!(
            valid_dir.exists(),
            "missing valid log dir for epoch {epoch}: {}",
            valid_dir.display()
        );

        // First epoch's logs remain unchanged.
        if epoch == 1 {
            let loss_file_train = train_dir.join("Loss.log");
            let loss_file_valid = valid_dir.join("Loss.log");

            let content_train = fs::read(loss_file_train).expect("Cannot read file.");
            let content_valid = fs::read(loss_file_valid).expect("Cannot read file.");

            assert_eq!(
                content_train, content_train_expected,
                "First epoch's logs should remain unchanged",
            );
            assert_eq!(
                content_valid, content_valid_expected,
                "First epoch's logs should remain unchanged",
            );
        }
    }
}
