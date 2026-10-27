mod common;

use std::sync::Arc;

use burn_core::{
    module::Param,
    tensor::{Device, Tensor, TensorReadError},
};
use burn_std::ExecutionError;
use burn_train::{
    EventProcessorFailure, EventProcessorTraining, ItemLazy, Learner, LearnerEvent, LearnerModel,
    MetricRegistration, OptimizerMetricRegistration, SupervisedLearningStrategy,
    SupervisedTraining, SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents,
    TrainingError, TrainingItem, TrainingStrategy, ValidLoader,
    logger::{FileMetricLogger, MetricLogger},
    metric::optimizer::{
        LearningRateMetric, OptimizerMetricContext, OptimizerStepMetric, OptimizerStepOutput,
    },
    metric::{
        LossMetric, MetricDefinition, MetricEntry, MetricId, MetricName, NumericAttributes,
        NumericEntry, SerializedEntry,
        store::{MetricsUpdate, NumericMetricUpdate, Split},
    },
};
use common::*;

// This entire extension uses public APIs: neither the observations nor the metric belong to Burn.
struct Observations {
    update_norm: Tensor<1>,
    fail_sync: bool,
}

impl ItemLazy for Observations {
    fn sync(self) -> Result<Self, ExecutionError> {
        if self.fail_sync {
            return Err(ExecutionError::with_context("observation sync failed"));
        }
        self.update_norm.device().flush()?;
        Ok(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    Sync,
    Missing,
    Update,
    Final,
}

struct ScalarMetric {
    name: MetricName,
    last: Option<f64>,
    failure: Failure,
    read: fn(&OptimizerStepOutput) -> Result<f64, TensorReadError>,
}

impl ScalarMetric {
    fn new(
        name: &str,
        failure: Failure,
        read: fn(&OptimizerStepOutput) -> Result<f64, TensorReadError>,
    ) -> Self {
        Self {
            name: Arc::new(name.into()),
            last: None,
            failure,
            read,
        }
    }

    fn entry(&self, value: NumericEntry) -> MetricsUpdate {
        MetricsUpdate::new(
            vec![],
            vec![NumericMetricUpdate::new(
                MetricEntry::new(
                    MetricId::new(self.name.clone()),
                    SerializedEntry::new(value.current().to_string(), value.serialize()),
                ),
                Some(value),
                None,
            )],
        )
    }
}

impl<M: LearnerModel> MetricRegistration<M> for ScalarMetric {
    fn register(self, training: SupervisedTraining<M>) -> SupervisedTraining<M> {
        OptimizerMetricRegistration::new(self).register(training)
    }
}

// Existing generic registration helpers and explicit type arguments need no extra parameter.
fn register<M: LearnerModel, Me: MetricRegistration<M>>(
    training: SupervisedTraining<M>,
    metrics: Me,
) -> SupervisedTraining<M> {
    training.metrics::<Me>(metrics)
}

impl OptimizerStepMetric for ScalarMetric {
    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn init(&mut self, _: &OptimizerMetricContext) -> Vec<MetricDefinition> {
        vec![MetricDefinition {
            metric_id: MetricId::new(self.name.clone()),
            name: self.name.clone(),
            description: None,
            attributes: NumericAttributes::default().into(),
        }]
    }

    fn clear(&mut self) {
        self.last = None;
    }

    fn update(&mut self, input: &OptimizerStepOutput) -> Result<MetricsUpdate, TensorReadError> {
        if self.failure == Failure::Update {
            return Err(ExecutionError::with_context("metric update failed").into());
        }
        let value = (self.read)(input)?;
        self.last = Some(value);
        Ok(self.entry(NumericEntry::Value(value)))
    }

    fn end_epoch(&mut self) -> Result<MetricsUpdate, TensorReadError> {
        if self.failure == Failure::Final {
            return Err(ExecutionError::with_context("metric finalization failed").into());
        }
        Ok(match self.last {
            Some(value) => self.entry(NumericEntry::Final(value)),
            None => MetricsUpdate::new(vec![], vec![]),
        })
    }
}

struct CustomStrategy(Failure);

impl SupervisedLearningStrategy<ToyModel> for CustomStrategy {
    fn fit(
        &self,
        components: TrainingComponents<ToyModel>,
        mut learner: Learner<ToyModel>,
        train: TrainLoader<ToyModel>,
        _: ValidLoader<ToyModel>,
        starting_epoch: usize,
    ) -> (ToyModel, SupervisedTrainingEventProcessor<ToyModel>) {
        let mut processor = components.event_processor;
        let interrupter = components.interrupter;
        for epoch in starting_epoch..=components.num_epochs {
            let mut iterator = train.iter();
            let mut progress = iterator.progress();
            interrupter.fail_on_error(processor.process_train(LearnerEvent::StartSplit {
                epoch_number: epoch,
                progress: progress.clone(),
            }));
            // A second, empty epoch exercises metric clearing in the real event processor.
            if epoch == starting_epoch {
                let mut iteration = 0;
                while let Some(batch) = iterator.next() {
                    iteration += 1;
                    let before = learner.model().weight.val().without_autodiff();
                    let output = learner.train_step(batch.unwrap());
                    learner.lr_step();
                    learner.optimizer_step(output.grads);
                    let after = learner.model().weight.val().without_autodiff();
                    let update_norm = (after - before).square().sum().sqrt();
                    let step = OptimizerStepOutput::new(learner.lr_current());
                    let step = if self.0 == Failure::Missing {
                        step
                    } else {
                        step.with_observation(Observations {
                            update_norm,
                            fail_sync: self.0 == Failure::Sync,
                        })
                    };
                    interrupter
                        .fail_on_error(processor.process_train(LearnerEvent::OptimizerStep(step)));
                    // Batch observations and optimizer observations share the same processor.
                    interrupter.fail_on_error(processor.process_train(
                        LearnerEvent::ProcessedItem(TrainingItem::new(
                            output.item,
                            iterator.progress(),
                            Some(iteration),
                            None,
                        )),
                    ));
                    interrupter.fail_on_error(processor.flush());
                    if interrupter.should_stop() {
                        break;
                    }
                }
                progress = iterator.progress();
            }
            interrupter
                .fail_on_error(processor.process_train(LearnerEvent::EndSplit { epoch, progress }));
            interrupter.fail_on_error(processor.flush());
            if interrupter.should_stop() {
                break;
            }
            interrupter.fail_on_error(processor.process_train(LearnerEvent::EndEpoch(epoch)));
        }
        (learner.model(), processor)
    }
}

#[test]
fn custom_metric_consumes_builtin_observations() {
    let device = Device::flex().autodiff();
    let (train, valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let result = SupervisedTraining::new(dir.path(), train, valid)
        .metrics::<OptimizerMetricRegistration<ScalarMetric>>(OptimizerMetricRegistration::new(
            ScalarMetric::new("Custom LR", Failure::None, |output| {
                Ok(output.learning_rate().base())
            }),
        ))
        .with_application_logger(None)
        .launch(make_learner(&device));
    assert!(result.error.is_none());
    let entries = FileMetricLogger::new(dir.path())
        .read_numeric("Custom LR", 1, &Split::Train)
        .unwrap();
    assert!(matches!(
        entries.as_slice(),
        [
            NumericEntry::Value(0.001),
            NumericEntry::Value(0.001),
            NumericEntry::Final(0.001)
        ]
    ));
}

#[rstest::rstest]
#[case(Failure::None)]
#[case(Failure::Sync)]
#[case(Failure::Missing)]
#[case(Failure::Update)]
#[case(Failure::Final)]
fn custom_observations_share_reporting_and_error_handling(#[case] failure: Failure) {
    let device = Device::flex().autodiff();
    let model = ToyModel {
        weight: Param::from_tensor(Tensor::from_data([[1.0, 2.0]], &device)),
    };
    let learner = Learner::new(model, burn_optim::SgdConfig::new().init(), 0.1);
    let (train, valid) = make_dataloaders();
    let dir = tempfile::tempdir().unwrap();
    let training = SupervisedTraining::new(dir.path(), train, valid)
        .with_training_strategy(TrainingStrategy::Custom(Arc::new(CustomStrategy(failure))))
        .num_epochs(2);
    let result = register(
        training,
        (
            LossMetric::new(),
            ScalarMetric::new("Update Norm", failure, |output| {
                let observation = output.observation::<Observations>().ok_or_else(|| {
                    ExecutionError::with_context("missing Update Norm observation")
                })?;
                observation.update_norm.clone().try_into_scalar::<f64>()
            }),
            LearningRateMetric::new(),
        ),
    )
    .with_application_logger(None)
    .launch(learner);
    assert!(result.interrupted.is_none());
    match failure {
        Failure::None => assert!(result.error.is_none()),
        _ => {
            let Some(TrainingError::EventProcessor(error)) = result.error.as_deref() else {
                panic!("expected metric failure");
            };
            match (&error.failures()[0], failure) {
                (EventProcessorFailure::Sync { split, .. }, Failure::Sync) => {
                    assert_eq!(*split, Split::Train)
                }
                (EventProcessorFailure::Metric(error), _) => {
                    assert_eq!(error.metric.as_str(), "Update Norm")
                }
                _ => panic!("wrong failure kind"),
            }
        }
    }
    let mut logger = FileMetricLogger::new(dir.path());
    let rates = logger
        .read_numeric("Learning Rate", 1, &Split::Train)
        .unwrap();
    // A failing metric must not suppress later metrics; a failed sync prevents all metrics.
    let updates = match failure {
        Failure::Sync => 0,
        Failure::Update | Failure::Missing => 1,
        _ => 2,
    };
    assert_eq!(rates.len(), if updates == 0 { 0 } else { updates + 1 });
    let norms = logger
        .read_numeric("Update Norm", 1, &Split::Train)
        .unwrap();
    match failure {
        Failure::None | Failure::Final => {
            assert_eq!(norms.len(), if failure == Failure::None { 3 } else { 2 });
            for (entry, expected) in norms[..2].iter().zip([0.05_f64.sqrt(), 0.0405_f64.sqrt()]) {
                assert!(
                    matches!(entry, NumericEntry::Value(value) if (*value - expected).abs() < 1e-6)
                );
            }
            if failure == Failure::None {
                assert!(
                    matches!(norms.last(), Some(NumericEntry::Final(value)) if (*value - 0.0405_f64.sqrt()).abs() < 1e-6)
                );
                assert_eq!(
                    logger.read_numeric("Loss", 1, &Split::Train).unwrap().len(),
                    3
                );
                for name in ["Update Norm", "Learning Rate"] {
                    assert!(
                        logger
                            .read_numeric(name, 2, &Split::Train)
                            .unwrap()
                            .is_empty()
                    );
                }
            }
        }
        _ => assert!(norms.is_empty()),
    }
}
