use crate::{
    ApplicationLoggerInstaller, Evaluator, FileApplicationLoggerInstaller, TestStep,
    TrainingInterrupter,
    evaluator::components::EvaluatorComponentTypesMarker,
    logger::FileMetricLogger,
    metric::{
        Adaptor, ItemLazy, Metric,
        processor::{AsyncProcessorEvaluation, FullEventProcessorEvaluation, MetricsEvaluation},
        store::{EventStoreClient, LogEventStore},
    },
    renderer::{MetricsRenderer, default_renderer},
};
use burn_core::{module::Module, prelude::Backend};
use std::{
    collections::BTreeSet,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

/// TODO: Docs
pub struct EvaluatorBuilder<B: Backend, TI, TO: ItemLazy> {
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    event_store: LogEventStore,
    summary_metrics: BTreeSet<String>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    interrupter: TrainingInterrupter,
    metrics: MetricsEvaluation<TO>,
    directory: PathBuf,
    summary: bool,
    _p: PhantomData<(B, TI, TO)>,
}

impl<B: Backend, TI, TO: ItemLazy + 'static> EvaluatorBuilder<B, TI, TO> {
    /// Creates a new learner builder.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("evaluator.log");

        Self {
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                experiment_log_file,
            ))),
            event_store: LogEventStore::default(),
            summary_metrics: Default::default(),
            renderer: None,
            interrupter: TrainingInterrupter::new(),
            summary: false,
            metrics: MetricsEvaluation::default(),
            directory,
            _p: PhantomData,
        }
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_numeric<Me: Metric + crate::metric::Numeric + 'static>(
        mut self,
        metric: Me,
    ) -> Self
    where
        <TO as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name());
        self.metrics.register_test_metric_numeric(metric);
        self
    }

    /// Replace the default CLI renderer with a custom one.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The custom renderer.
    pub fn renderer(mut self, renderer: Option<Box<dyn MetricsRenderer + 'static>>) -> Self {
        self.renderer = renderer;
        self
    }

    /// Enable the evaluation summary report.
    ///
    /// The summary will be displayed at the end of `.eval()`.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }

    /// Builds the evaluator.
    pub fn build<M>(
        mut self,
        model: M,
    ) -> Evaluator<
        EvaluatorComponentTypesMarker<
            B,
            M,
            AsyncProcessorEvaluation<FullEventProcessorEvaluation<TO>>,
            TI,
            TO,
        >,
    >
    where
        TI: Send + 'static,
        M: Module<B> + TestStep<TI, TO> + core::fmt::Display + 'static,
    {
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), None));

        self.event_store
            .register_logger_test(FileMetricLogger::new_eval(self.directory.join("test")));
        let event_store = Arc::new(EventStoreClient::new(self.event_store));

        let event_processor = AsyncProcessorEvaluation::new(FullEventProcessorEvaluation::new(
            self.metrics,
            renderer,
            event_store.clone(),
        ));

        Evaluator {
            model,
            interrupter: self.interrupter,
            event_processor,
            event_store,
        }
    }
}
