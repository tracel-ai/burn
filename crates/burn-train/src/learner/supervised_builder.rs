use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components_v2::{
    InputTrainV2, InputValidV2, LearnerComponentTypesV2, LearnerComponentsMarkerV2,
    LearningDataMarkerV2, LearningDataV2, OutputTrainV2, OutputValidV2, TrainBackendV2,
    TrainLoaderV2, TrainModelV2, TrainOptmizerV2, TrainSchedulerV2, ValidLoaderV2,
};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::Interrupter;
use crate::learner::base_v2::LearnerV2;
use crate::learner::paradigms::{
    SingleDeviceLearningStrategyV2, SupervisedLearningStrategy, TrainingStrategy,
};
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{
    AsyncProcessorTraining, FullEventProcessorTraining, ItemLazy, MetricsTraining,
};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::single::SingleDeviceLearningStrategy;
use crate::{
    ApplicationLoggerInstaller, EarlyStoppingStrategyRef, FileApplicationLoggerInstaller,
    LearnerSummaryConfig, LearningParadigm, ModelRecordTrain, OptimizerRecordTrain,
    ParadigmComponents, SchedulerRecordTrain, SupervisedComponents, TrainStep,
    TrainingCheckpointer, TrainingResult, ValidStep,
};
use burn_core::module::{AutodiffModule, Module};
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;

// pub trait SupervisedParadigmBound = ParadigmComponents<
//         CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
//         EventProcessor = AsyncProcessorTraining<FullEventProcessorTraining<ParadigmOutputTrain<PC>, ParadigmOutputValid<PC>>>,
//         LearnerComponents = LearnerComponentTypesV2<
//                 CheckpointerModel = AsyncCheckpointer<
//                     <ParadigmModel<PC> as Module<ParadigmBackendTrain<PC>>>::Record,
//                     ParadigmBackendTrain<PC>,
//                 >,
//                 CheckpointerOptimizer = AsyncCheckpointer<
//                     <ParadigmOptimizer<PC> as Optimizer<ParadigmModel<PC>, ParadigmBackendTrain<PC>>>::Record,
//                     ParadigmBackendTrain<PC>,
//                 >,
//                 CheckpointerLrScheduler = AsyncCheckpointer<
//                     <ParadigmScheduler<PC> as LrScheduler>::Record<ParadigmBackendTrain<PC>>,
//                     ParadigmBackendTrain<PC>,
//                 >,
//             >>;

// pub struct SupervisedComponentsMarker<PC: ParadigmComponents> {
//     _components: PhantomData<PC>,
// }

// impl<PC> ParadigmComponents for SupervisedComponentsMarker<PC>
// // impl<B, LR, M, O> LearnerComponentTypesV2 for LearnerComponentsMarkerV2<B, LR, M, O>
// where
//     PC: ParadigmComponents<CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
//         EventProcessor = AsyncProcessorTraining<FullEventProcessorTraining<ParadigmOutputTrain<PC>, ParadigmOutputValid<PC>>>>,
//     PC::LearnerComponents: LearnerComponentTypesV2<
//             CheckpointerModel = AsyncCheckpointer<
//                 <ParadigmModel<PC> as Module<ParadigmBackendTrain<PC>>>::Record,
//                 ParadigmBackendTrain<PC>,
//             >,
//             CheckpointerOptimizer = AsyncCheckpointer<
//                 <ParadigmOptimizer<PC> as Optimizer<ParadigmModel<PC>, ParadigmBackendTrain<PC>>>::Record,
//                 ParadigmBackendTrain<PC>,
//             >,
//             CheckpointerLrScheduler = AsyncCheckpointer<
//                 <ParadigmScheduler<PC> as LrScheduler>::Record<ParadigmBackendTrain<PC>>,
//                 ParadigmBackendTrain<PC>,
//             >,
//         >,
//     <PC::LearnerComponents as LearnerComponentTypesV2>::Model:
//         TrainStep<InputTrainV2<PC::LearningData>, OutputTrainV2<PC::LearningData>>,
//     <PC::LearnerComponents as LearnerComponentTypesV2>::InnerModel:
//         ValidStep<InputValidV2<PC::LearningData>, OutputValidV2<PC::LearningData>>,
// {
//     type LearnerComponents = PC::LearnerComponents;

//     type LearningData = PC::LearningData;

//     type EventProcessor = PC::EventProcessor;

//     type CheckpointerStrategy = PC::CheckpointerStrategy;
// }

// pub struct SupervisedLearningMarker<PC: ParadigmComponents>
// where
//     Self: LearnerComponentTypesV2<
//             CheckpointerModel = AsyncCheckpointer<
//                 <TrainModelV2<Self> as Module<TrainBackendV2<Self>>>::Record,
//                 TrainBackendV2<Self>,
//             >,
//             CheckpointerOptimizer = AsyncCheckpointer<
//                 <TrainOptmizerV2<Self> as Optimizer<TrainModelV2<Self>, TrainBackendV2<Self>>>::Record,
//                 TrainBackendV2<Self>,
//             >,
//             CheckpointerLrScheduler = AsyncCheckpointer<
//                 <TrainSchedulerV2<Self> as LrScheduler>::Record<TrainBackendV2<Self>>,
//                 TrainBackendV2<Self>,
//             >,
//         >,
//     // <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record: 'static,
//     // <Self::LrScheduler as LrScheduler>::Record<Self::Backend>: 'static ,
//     // Self::Model:
//     //     TrainStep<InputTrainV2<LD>, OutputTrainV2<LD>>,
//     // Self::InnerModel:
//     //     ValidStep<InputValidV2<LD>, OutputValidV2<LD>>,{
//     {
//     _components: PhantomData<PC>
// }

// type SupervisedModel<PC> = <SupervisedLearningMarker<PC>::LearnerComponents as LearnerComponentTypesV2>::Model;

// pub struct SupervisedLearnerComponentsMarker<LD: LearningDataV2>{
//     _components: PhantomData<LD>
// }

// impl<LD> LearnerComponentTypesV2 for SupervisedLearnerComponentsMarker<LD>
// where
//     LD: LearningDataV2,
//     Self: LearnerComponentTypesV2<
//             CheckpointerModel = AsyncCheckpointer<
//                 <TrainModelV2<Self> as Module<TrainBackendV2<Self>>>::Record,
//                 TrainBackendV2<Self>,
//             >,
//             CheckpointerOptimizer = AsyncCheckpointer<
//                 <TrainOptmizerV2<Self> as Optimizer<TrainModelV2<Self>, TrainBackendV2<Self>>>::Record,
//                 TrainBackendV2<Self>,
//             >,
//             CheckpointerLrScheduler = AsyncCheckpointer<
//                 <TrainSchedulerV2<Self> as LrScheduler>::Record<TrainBackendV2<Self>>,
//                 TrainBackendV2<Self>,
//             >,
//         >,
//     <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record: 'static,
//     <Self::LrScheduler as LrScheduler>::Record<Self::Backend>: 'static ,
//     Self::Model:
//         TrainStep<InputTrainV2<LD>, OutputTrainV2<LD>>,
//     Self::InnerModel:
//         ValidStep<InputValidV2<LD>, OutputValidV2<LD>>,
// {
//     type Backend = Self::Backend;
//     type LrScheduler = Self::LrScheduler;
//     type Model = Self::Model;
//     type InnerModel = Self::InnerModel;
//     type Optimizer = Self::Optimizer;
//     type CheckpointerModel = Self::CheckpointerModel;
//     type CheckpointerOptimizer = Self::CheckpointerOptimizer;
//     type CheckpointerLrScheduler = Self::CheckpointerLrScheduler;
// }

// pub trait SupervisedComponentsBound: Sized
// where
//     Self: ParadigmComponents<
//         CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
//         EventProcessor = AsyncProcessorTraining<
//             FullEventProcessorTraining<
//                 ParadigmOutputTrain<Self>,
//                 ParadigmOutputValid<Self>
//             >
//         >,
//         // LearnerComponents = SupervisedLearnerComponentsMarker<Self::LearningData>
//     >,
//     // <Self as ParadigmComponents>::LearnerComponents: LearnerComponentTypesV2<
//     //     CheckpointerModel = AsyncCheckpointer<
//     //         <ParadigmModel<Self> as Module<ParadigmBackendTrain<Self>>>::Record,
//     //         ParadigmBackendTrain<Self>
//     //     >,
//     //     CheckpointerOptimizer = AsyncCheckpointer<
//     //         <ParadigmOptimizer<Self> as Optimizer<
//     //             ParadigmModel<Self>,
//     //             ParadigmBackendTrain<Self>
//     //         >>::Record,
//     //         ParadigmBackendTrain<Self>
//     //     >,
//     //     CheckpointerLrScheduler = AsyncCheckpointer<
//     //         <ParadigmScheduler<Self> as LrScheduler>::Record<ParadigmBackendTrain<Self>>,
//     //         ParadigmBackendTrain<Self>
//     //     >,
//     // >,
//     // < Self::LearnerComponents as LearnerComponentTypesV2 >::Model:
//     //     TrainStep<
//     //         ParadigmInputTrain<Self>,
//     //         ParadigmOutputTrain<Self>
//     //     >,
//     // < Self::LearnerComponents as LearnerComponentTypesV2 >::InnerModel:
//     //     ValidStep<
//     //         ParadigmInputValid<Self>,
//     //         ParadigmOutputValid<Self>,
//     //     >,
// {}

/// Struct to configure and create a [learner](Learner).
///
/// The generics components of the builder should probably not be set manually, as they are
/// optimized for Rust type inference.

// pub struct SupervisedTraining<B, M, O, S, TI, VI, TO, VO>
// where
//     B: AutodiffBackend,
//     M: AutodiffModule<B> + LearningModel + TrainStep<TI, TO> + core::fmt::Display + 'static,
//     M::InnerModule: ValidStep<VI, VO>,
//     O: Optimizer<M, B> + 'static,
//     S: LrScheduler + 'static,
//     TI: Send + 'static,
//     VI: Send + 'static,
//     TO: ItemLazy + 'static,
//     VO: ItemLazy + 'static,
// pub struct SupervisedTraining<SC: SupervisedComponents>
// {
//     // Not that complex and very convenient when the traits are
//     // already constrained correctly. Extracting in another type
//     // would be more complex.
//     #[allow(clippy::type_complexity)] checkpointers: Option<(
//         AsyncCheckpointer<ModelRecordTrain<SC::LC>, SC::Backend>,
//         AsyncCheckpointer<OptimizerRecordTrain<SC::LC>, SC::Backend>,
//         AsyncCheckpointer<SchedulerRecordTrain<SC::LC>, SC::Backend>,
//     )>,
//     num_epochs: usize,
//     checkpoint: Option<usize>,
//     directory: PathBuf,
//     grad_accumulation: Option<usize>,
//     renderer: Option<Box<dyn MetricsRenderer + 'static>>,
//     metrics: MetricsTraining<, ParadigmOutputValid<PC>>,
//     event_store: LogEventStore,
//     interrupter: Interrupter,
//     tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
//     checkpointer_strategy: Box<dyn CheckpointingStrategy>,
//     early_stopping: Option<EarlyStoppingStrategyRef>,
//     training_strategy: TrainingStrategy<PC::LearnerComponents>,
//     dataloader_train: TrainLoaderV2<PC::LearnerComponents, PC::LearningData>,
//     dataloader_valid: ValidLoaderV2<PC::LearnerComponents, PC::LearningData>,
//     // learner: LearnerV2<LC<B, S, M, O>>,
//     // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
//     summary_metrics: BTreeSet<String>,
//     summary: bool,
// }

pub struct SupervisedTraining<SC: SupervisedComponents> {
    // Not that complex and very convenient when the traits are
    // already constrained correctly. Extracting in another type
    // would be more complex.
    // #[allow(clippy::type_complexity)]
    // checkpointers: Option<(
    //     AsyncCheckpointer<ModelRecordTrain<SC::LC>, SC::Backend>,
    //     AsyncCheckpointer<OptimizerRecordTrain<SC::LC>, SC::Backend>,
    //     AsyncCheckpointer<SchedulerRecordTrain<SC::LC>, SC::Backend>,
    // )>,
    #[allow(clippy::type_complexity)]
    checkpointers: Option<(
        AsyncCheckpointer<<SC::Model as Module<SC::Backend>>::Record, SC::Backend>,
        AsyncCheckpointer<
            <SC::Optimizer as Optimizer<SC::Model, SC::Backend>>::Record,
            SC::Backend,
        >,
        AsyncCheckpointer<<SC::LrScheduler as LrScheduler>::Record<SC::Backend>, SC::Backend>,
    )>,
    num_epochs: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: MetricsTraining<OutputTrainV2<SC::LD>, OutputValidV2<SC::LD>>,
    event_store: LogEventStore,
    interrupter: Interrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<EarlyStoppingStrategyRef>,
    training_strategy: TrainingStrategy<SC::LC>,
    dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
    dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
    learner: LearnerV2<SC::LC>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
}

type LC<B, S, M, O> = LearnerComponentsMarkerV2<
    B,
    S,
    M,
    O,
    AsyncCheckpointer<<M as Module<B>>::Record, B>,
    AsyncCheckpointer<<O as Optimizer<M, B>>::Record, B>,
    AsyncCheckpointer<<S as LrScheduler>::Record<B>, B>,
    // Box<dyn CheckpointingStrategy>,
>;

type LD<TI, VI, TO, VO> = LearningDataMarkerV2<TI, VI, TO, VO>;

// impl<B, M, O, S, TI, VI, TO, VO> SupervisedTraining<B, M, O, S, TI, VI, TO, VO>
// where
//     B: AutodiffBackend,
//     M: AutodiffModule<B> + LearningModel + TrainStep<TI, TO> + core::fmt::Display + 'static,
//     M::InnerModule: ValidStep<VI, VO>,
//     O: Optimizer<M, B>,
//     S: LrScheduler,
//     TI: Send + 'static,
//     VI: Send + 'static,
//     TO: ItemLazy + 'static,
//     VO: ItemLazy + 'static,
impl<SC: SupervisedComponents> SupervisedTraining<SC>
// where
// PC: SupervisedComponentsBound,
// PC::LearnerComponents: LearnerComponentTypesV2<
//         CheckpointerModel = AsyncCheckpointer<
//             <ParadigmModel<PC> as Module<ParadigmBackendTrain<PC>>>::Record,
//             ParadigmBackendTrain<PC>,
//         >,
//         CheckpointerOptimizer = AsyncCheckpointer<
//             <ParadigmOptimizer<PC> as Optimizer<ParadigmModel<PC>, ParadigmBackendTrain<PC>>>::Record,
//             ParadigmBackendTrain<PC>,
//         >,
//         CheckpointerLrScheduler = AsyncCheckpointer<
//             <ParadigmScheduler<PC> as LrScheduler>::Record<ParadigmBackendTrain<PC>>,
//             ParadigmBackendTrain<PC>,
//         >,
//     >,
// <PC::LearnerComponents as LearnerComponentTypesV2>::Model:
//     TrainStep<InputTrainV2<PC::LearningData>, OutputTrainV2<PC::LearningData>>,
// <PC::LearnerComponents as LearnerComponentTypesV2>::InnerModel:
//     ValidStep<InputValidV2<PC::LearningData>, OutputValidV2<PC::LearningData>>,
{
    /// Creates a new learner builder.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(
        directory: impl AsRef<Path>,
        dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
        dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
        learner: LearnerV2<SC::LC>,
    ) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            num_epochs: 1,
            checkpoint: None,
            checkpointers: None,
            directory,
            grad_accumulation: None,
            metrics: MetricsTraining::default(),
            event_store: LogEventStore::default(),
            renderer: None,
            interrupter: Interrupter::new(),
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                experiment_log_file,
            ))),
            checkpointer_strategy: Box::new(
                ComposedCheckpointingStrategy::builder()
                    .add(KeepLastNCheckpoints::new(2))
                    .add(MetricCheckpointingStrategy::new(
                        &LossMetric::<SC::Backend>::new(), // default to valid loss
                        Aggregate::Mean,
                        Direction::Lowest,
                        Split::Valid,
                    ))
                    .build(),
            ),
            early_stopping: None,
            training_strategy: TrainingStrategy::SingleDevice(Default::default()),
            summary_metrics: BTreeSet::new(),
            summary: false,
            dataloader_train,
            dataloader_valid,
            learner,
        }
    }

    /// Replace the default metric loggers with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `logger` - The training logger.
    pub fn with_metric_logger<ML>(mut self, logger: ML) -> Self
    where
        ML: MetricLogger + 'static,
    {
        self.event_store.register_logger(logger);
        self
    }

    /// Update the checkpointing_strategy.
    pub fn with_checkpointing_strategy(
        mut self,
        strategy: <SC::PC as ParadigmComponents>::CheckpointerStrategy,
    ) -> Self
    where
        // CS: CheckpointingStrategy + 'static,
        <SC::PC as ParadigmComponents>::CheckpointerStrategy: 'static,
    {
        self.checkpointer_strategy = Box::new(strategy);
        self
    }

    /// Replace the default CLI renderer with a custom one.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The custom renderer.
    pub fn renderer<MR>(mut self, renderer: MR) -> Self
    where
        MR: MetricsRenderer + 'static,
    {
        self.renderer = Some(Box::new(renderer));
        self
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics<Me: MetricRegistrationV2<SC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics_text<Me: TextMetricRegistrationV2<SC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register a training metric.
    pub fn metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputTrainV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_train_metric(metric);
        self
    }

    /// Register a validation metric.
    pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputValidV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_valid_metric(metric);
        self
    }

    /// Enable gradients accumulation.
    ///
    /// # Notes
    ///
    /// When you enable gradients accumulation, the gradients object used by the optimizer will be
    /// the sum of all gradients generated by each backward pass. It might be a good idea to
    /// reduce the learning to compensate.
    ///
    /// The effect is similar to increasing the `batch size` and the `learning rate` by the `accumulation`
    /// amount.
    pub fn grads_accumulation(mut self, accumulation: usize) -> Self {
        self.grad_accumulation = Some(accumulation);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) training [metric](Metric).
    pub fn metric_train_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <OutputTrainV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_valid_numeric<Me: Metric + Numeric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputValidV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_valid_metric_numeric(metric);
        self
    }

    /// The number of epochs the training should last.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// The epoch from which the training must resume.
    pub fn checkpoint(mut self, checkpoint: usize) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Provides a handle that can be used to interrupt training.
    pub fn interrupter(&self) -> Interrupter {
        self.interrupter.clone()
    }

    /// Override the handle for stopping training with an externally provided handle
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.interrupter = interrupter;
        self
    }

    /// Register an [early stopping strategy](EarlyStoppingStrategy) to stop the training when the
    /// conditions are meet.
    pub fn early_stopping<Strategy>(mut self, strategy: Strategy) -> Self
    where
        Strategy: EarlyStoppingStrategy + Clone + Send + Sync + 'static,
    {
        self.early_stopping = Some(Box::new(strategy));
        self
    }

    /// By default, Rust logs are captured and written into
    /// `experiment.log`. If disabled, standard Rust log handling
    /// will apply.
    pub fn with_application_logger(
        mut self,
        logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    ) -> Self {
        self.tracing_logger = logger;
        self
    }

    /// Register a checkpointer that will save the [optimizer](Optimizer), the
    /// [model](AutodiffModule) and the [scheduler](LrScheduler) to different files.
    pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
    where
        FR: FileRecorder<SC::Backend> + 'static,
        FR: FileRecorder<<SC::Backend as AutodiffBackend>::InnerBackend> + 'static,
        <SC::Optimizer as Optimizer<SC::Model, SC::Backend>>::Record: 'static,
        <SC::LrScheduler as LrScheduler>::Record<SC::Backend>: 'static,
    {
        let checkpoint_dir = self.directory.join("checkpoint");
        let checkpointer_model = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model");
        let checkpointer_optimizer =
            FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim");
        let checkpointer_scheduler: FileCheckpointer<FR> =
            FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler");

        self.checkpointers = Some((
            AsyncCheckpointer::new(checkpointer_model),
            AsyncCheckpointer::new(checkpointer_optimizer),
            AsyncCheckpointer::new(checkpointer_scheduler),
        ));

        self
    }

    /// Enable the training summary report.
    ///
    /// The summary will be displayed after `.fit()`, when the renderer is dropped.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }

    // /// Create the [learner](Learner) from a [model](AutodiffModule) and an [optimizer](Optimizer).
    // /// The [learning rate scheduler](LrScheduler) can also be a simple
    // /// [learning rate](burn_optim::LearningRate).
    // #[allow(clippy::type_complexity)] // The goal for the builder is to handle all types and
    // // creates a clean learner.
    // pub fn run(
    //     mut self,
    //     learner: LearnerV2<LC<B, S, M, O>>,
    //     // training_strategy: LearningStrategy<LC<B, S, M, O, TO, VO, TI, VI>>,
    // ) -> TrainingResult<<LC<B, S, M, O> as LearnerComponentTypesV2>::InnerModel>
    // where
    //     M::Record: 'static,
    //     O::Record: 'static,
    //     S::Record<B>: 'static,
    // {
    //     if self.tracing_logger.is_some()
    //         && let Err(e) = self.tracing_logger.as_ref().unwrap().install()
    //     {
    //         log::warn!("Failed to install the experiment logger: {e}");
    //     }
    //     let renderer = self
    //         .renderer
    //         .unwrap_or_else(|| default_renderer(self.interrupter.clone(), self.checkpoint));

    //     if !self.event_store.has_loggers() {
    //         self.event_store
    //             .register_logger(FileMetricLogger::new(self.directory.clone()));
    //     }

    //     let event_store = Arc::new(EventStoreClient::new(self.event_store));
    //     let event_processor = AsyncProcessorTraining::new(FullEventProcessorTraining::new(
    //         self.metrics,
    //         renderer,
    //         event_store.clone(),
    //     ));

    //     let checkpointer = self.checkpointers.map(|(model, optim, scheduler)| {
    //         LearnerCheckpointerV2::<LC<B, S, M, O>, Box<dyn CheckpointingStrategy>>::new(
    //             model,
    //             optim,
    //             scheduler,
    //             self.checkpointer_strategy,
    //         )
    //     });

    //     let summary = if self.summary {
    //         Some(LearnerSummaryConfig {
    //             directory: self.directory,
    //             metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
    //         })
    //     } else {
    //         None
    //     };

    //     // let training_strategy = Self::prepare_learning_strategy(training_strategy);

    //     // Learner {
    //     //     model,
    //     //     optim,
    //     //     lr_scheduler,
    //     //     checkpointer,
    //     //     num_epochs: self.num_epochs,
    //     //     event_processor,
    //     //     event_store,
    //     //     checkpoint: self.checkpoint,
    //     //     grad_accumulation: self.grad_accumulation,
    //     //     training_strategy,
    //     //     interrupter: self.interrupter,
    //     //     early_stopping: self.early_stopping,
    //     //     summary,
    //     // }
    //     TrainingResult {
    //         model: learner.model.valid(),
    //         renderer: event_processor.renderer(),
    //     }
    // }

    // #[allow(clippy::type_complexity)]
    // fn prepare_learning_strategy(
    //     training_strategy: LearningStrategy<LC<B, S, M, O, TO, VO, TI, VI>>,
    // ) -> LearningStrategy<LC<B, S, M, O, TO, VO, TI, VI>>
    // where
    //     M::Record: 'static,
    //     O::Record: 'static,
    //     S::Record<B>: 'static,
    // {
    //     if let LearningStrategy::MultiDevice(devices, _) = training_strategy&
    //         && devices.len() == 1
    //     {
    //         return LearningStrategy::SingleDevice(devices[0].clone());
    //     }

    //    training_strategy
    // }
}

pub struct TrainingComponents<SC: SupervisedComponents> {
    pub checkpoint: Option<usize>,
    pub checkpointer: Option<TrainingCheckpointer<SC::LC, SC::PC>>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// An [EventProcessor](LearnerComponentTypesV2::EventProcessor) that processes events happening during training and validation.
    pub event_processor: <SC::PC as ParadigmComponents>::EventProcessor,
    // pub event_processor: AsyncProcessorTraining<
    //     FullEventProcessorTraining<
    //         OutputTrainV2<PC::LearningData>,
    //         OutputValidV2<PC::LearningData>,
    //     >,
    // >,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    pub num_epochs: usize,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    pub summary: Option<LearnerSummaryConfig>,
}

// impl<PC> LearningParadigm<PC>
//     for SupervisedTraining<
//         // ParadigmBackendTrain<PC>,
//         // ParadigmModel<PC>,
//         // ParadigmOptimizer<PC>,
//         // ParadigmScheduler<PC>,
//         // ParadigmInputTrain<PC>,
//         // ParadigmInputValid<PC>,
//         // ParadigmOutputTrain<PC>,
//         // ParadigmOutputValid<PC>,
//         PC,
//     >
// where
// PC: ParadigmComponents<CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
// EventProcessor = AsyncProcessorTraining<FullEventProcessorTraining<ParadigmOutputTrain<PC>, ParadigmOutputValid<PC>>>>,
// PC: SupervisedComponentsBound,
// PC::LearnerComponents: LearnerComponentTypesV2<
//         CheckpointerModel = AsyncCheckpointer<
//             <ParadigmModel<PC> as Module<ParadigmBackendTrain<PC>>>::Record,
//             ParadigmBackendTrain<PC>,
//         >,
//         CheckpointerOptimizer = AsyncCheckpointer<
//             <ParadigmOptimizer<PC> as Optimizer<ParadigmModel<PC>, ParadigmBackendTrain<PC>>>::Record,
//             ParadigmBackendTrain<PC>,
//         >,
//         CheckpointerLrScheduler = AsyncCheckpointer<
//             <ParadigmScheduler<PC> as LrScheduler>::Record<ParadigmBackendTrain<PC>>,
//             ParadigmBackendTrain<PC>,
//         >,
//     >,
// <PC::LearnerComponents as LearnerComponentTypesV2>::Model:
//     TrainStep<InputTrainV2<PC::LearningData>, OutputTrainV2<PC::LearningData>>,
// <PC::LearnerComponents as LearnerComponentTypesV2>::InnerModel:
//     ValidStep<InputValidV2<PC::LearningData>, OutputValidV2<PC::LearningData>>,
impl<SC: SupervisedComponents> LearningParadigm<SC::LC> for SupervisedTraining<SC> {
    fn train(mut self) -> TrainingResult<SC::InnerModel> {
        if self.tracing_logger.is_some()
            && let Err(e) = self.tracing_logger.as_ref().unwrap().install()
        {
            log::warn!("Failed to install the experiment logger: {e}");
        }
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), self.checkpoint));

        if !self.event_store.has_loggers() {
            self.event_store
                .register_logger(FileMetricLogger::new(self.directory.clone()));
        }

        let event_store = Arc::new(EventStoreClient::new(self.event_store));
        let event_processor = AsyncProcessorTraining::new(FullEventProcessorTraining::new(
            self.metrics,
            renderer,
            event_store.clone(),
        ));

        let checkpointer = self.checkpointers.map(|(model, optim, scheduler)| {
            TrainingCheckpointer::new(model, optim, scheduler, self.checkpointer_strategy)
        });

        let summary = if self.summary {
            Some(LearnerSummaryConfig {
                directory: self.directory,
                metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
            })
        } else {
            None
        };

        let components = TrainingComponents {
            checkpoint: self.checkpoint,
            checkpointer,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            event_processor,
            event_store,
            num_epochs: self.num_epochs,
            grad_accumulation: self.grad_accumulation,
            summary,
        };

        let learner = self.learner;

        match self.training_strategy {
            TrainingStrategy::SingleDevice(device) => {
                let single_device: SingleDeviceLearningStrategyV2<SC> =
                    SingleDeviceLearningStrategyV2::new(device);
                single_device.train(
                    learner,
                    self.dataloader_train,
                    self.dataloader_valid,
                    components,
                )
            }
        }

        // self.fit(
        //     learner,
        //     self.dataloader_train.clone(),
        //     self.dataloader_valid.clone(),
        //     self.strategy,
        // )
    }
}

/// Trait to fake variadic generics.
// pub trait MetricRegistrationV2<B, M, O, S, TI, VI, TO, VO>: Sized
pub trait MetricRegistrationV2<SC: SupervisedComponents>: Sized {
    // /// Register the metrics.
    // fn register(
    //     self,
    //     builder: SupervisedTraining<B, M, O, S, TI, VI, TO, VO>,
    // ) -> SupervisedTraining<B, M, O, S, TI, VI, TO, VO>;

    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<SC>) -> SupervisedTraining<SC>;
}

/// Trait to fake variadic generics.
// pub trait TextMetricRegistrationV2<B, M, O, S, TI, VI, TO, VO>: Sized
// where
//     B: AutodiffBackend,
//     M: AutodiffModule<B> + LearningModel + TrainStep<TI, TO> + core::fmt::Display + 'static,
//     M::InnerModule: ValidStep<VI, VO>,
//     O: Optimizer<M, B>,
//     S: LrScheduler,
//     TI: Send + 'static,
//     VI: Send + 'static,
//     TO: ItemLazy + 'static,
//     VO: ItemLazy + 'static,
// {
//     /// Register the metrics.
//     fn register(
//         self,
//         builder: SupervisedTraining<B, M, O, S, TI, VI, TO, VO>,
//     ) -> SupervisedTraining<B, M, O, S, TI, VI, TO, VO>;
// }

/// Trait to fake variadic generics.
pub trait TextMetricRegistrationV2<SC: SupervisedComponents>: Sized {
    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<SC>) -> SupervisedTraining<SC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        // impl<$($M,)* B, M, O, S, TI, VI, TO, VO> TextMetricRegistrationV2<B, M, O, S, TI, VI, TO, VO> for ($($M,)*)
        // where
        //     B: AutodiffBackend,
        //     M: AutodiffModule<B> + LearningModel + TrainStep<TI, TO> + core::fmt::Display + 'static,
        //     M::InnerModule: ValidStep<VI, VO>,
        //     O: Optimizer<M, B>,
        //     S: LrScheduler,
        //     TI: Send + 'static,
        //     VI: Send + 'static,
        //     TO: ItemLazy + 'static,
        //     VO: ItemLazy + 'static,
        //     $(TO::ItemSync: Adaptor<$M::Input>,)*
        //     $(VO::ItemSync: Adaptor<$M::Input>,)*
        //     $($M: Metric + 'static,)*
        // {
        //     #[allow(non_snake_case)]
        //     fn register(
        //         self,
        //         builder: SupervisedTraining<B, M, O, S, TI, VI, TO, VO>,
        //     ) -> SupervisedTraining<B, M, O, S, TI, VI, TO, VO> {
        //         let ($($M,)*) = self;
        //         $(let builder = builder.metric_train($M.clone());)*
        //         $(let builder = builder.metric_valid($M);)*
        //         builder
        //     }
        // }

        impl<$($M,)* SC: SupervisedComponents> TextMetricRegistrationV2<SC> for ($($M,)*)
        where
            $(<OutputTrainV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValidV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<SC>,
            ) -> SupervisedTraining<SC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train($M.clone());)*
                $(let builder = builder.metric_valid($M);)*
                builder
            }
        }

        // impl<$($M,)* B, M, O, S, TI, VI, TO, VO> MetricRegistrationV2  <B, M, O, S, TI, VI, TO, VO> for ($($M,)*)
        // where
        //     B: AutodiffBackend,
        //     M: AutodiffModule<B> + LearningModel + TrainStep<TI, TO> + core::fmt::Display + 'static,
        //     M::InnerModule: ValidStep<VI, VO>,
        //     O: Optimizer<M, B>,
        //     S: LrScheduler,
        //     TI: Send + 'static,
        //     VI: Send + 'static,
        //     TO: ItemLazy + 'static,
        //     VO: ItemLazy + 'static,
        //     $(TO::ItemSync: Adaptor<$M::Input>,)*
        //     $(VO::ItemSync: Adaptor<$M::Input>,)*
        //     $($M: Metric + Numeric + 'static,)*
        // {
        //     #[allow(non_snake_case)]
        //     fn register(
        //         self,
        //         builder: SupervisedTraining<B, M, O, S, TI, VI, TO, VO>,
        //     ) -> SupervisedTraining<B, M, O, S, TI, VI, TO, VO> {
        //         let ($($M,)*) = self;
        //         $(let builder = builder.metric_train_numeric($M.clone());)*
        //         $(let builder = builder.metric_valid_numeric($M);)*
        //         builder
        //     }
        // }

        impl<$($M,)* SC: SupervisedComponents> MetricRegistrationV2<SC> for ($($M,)*)
        where
            $(<OutputTrainV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValidV2<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<SC>,
            ) -> SupervisedTraining<SC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train_numeric($M.clone());)*
                $(let builder = builder.metric_valid_numeric($M);)*
                builder
            }
        }
    };
}

gen_tuple!(M1);
gen_tuple!(M1, M2);
gen_tuple!(M1, M2, M3);
gen_tuple!(M1, M2, M3, M4);
gen_tuple!(M1, M2, M3, M4, M5);
gen_tuple!(M1, M2, M3, M4, M5, M6);
