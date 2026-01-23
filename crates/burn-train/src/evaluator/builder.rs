use crate::{
    ApplicationLoggerInstaller, Evaluator, FileApplicationLoggerInstaller, InferenceStep,
    Interrupter, TestOutput,
    evaluator::components::{EvaluatorComponentTypes, EvaluatorComponentTypesMarker},
    logger::FileMetricLogger,
    metric::{
        Adaptor, ItemLazy, Metric, Numeric,
        processor::{AsyncProcessorEvaluation, FullEventProcessorEvaluation, MetricsEvaluation},
        store::{EventStoreClient, LogEventStore},
    },
    renderer::{MetricsRenderer, default_renderer},
};
use burn_core::{module::Module, prelude::Backend};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Struct to configure and create an [evaluator](Evaluator).
///
/// The generics components of the builder should probably not be set manually, as they are
/// optimized for Rust type inference.
pub struct EvaluatorBuilder<EC: EvaluatorComponentTypes> {
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    event_store: LogEventStore,
    summary_metrics: BTreeSet<String>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    interrupter: Interrupter,
    metrics: MetricsEvaluation<TestOutput<EC>>,
    directory: PathBuf,
    summary: bool,
}

impl<B, M> EvaluatorBuilder<EvaluatorComponentTypesMarker<B, M>>
where
    B: Backend,
    M: Module<B> + InferenceStep + core::fmt::Display + 'static,
{
    /// Creates a new evaluator builder.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let log_file = directory.join("evaluation.log");

        Self {
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(log_file))),
            event_store: LogEventStore::default(),
            summary_metrics: Default::default(),
            renderer: None,
            interrupter: Interrupter::new(),
            summary: false,
            metrics: MetricsEvaluation::default(),
            directory,
        }
    }
}

impl<EC: EvaluatorComponentTypes> EvaluatorBuilder<EC> {
    /// Registers [numeric](crate::metric::Numeric) test [metrics](Metric).
    pub fn metrics<Me: EvalMetricRegistration<EC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Registers text [metrics](Metric).
    pub fn metrics_text<Me: EvalTextMetricRegistration<EC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// By default, Rust logs are captured and written into
    /// `evaluation.log`. If disabled, standard Rust log handling
    /// will apply.
    pub fn with_application_logger(
        mut self,
        logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    ) -> Self {
        self.tracing_logger = logger;
        self
    }

    /// Register a [numeric](crate::metric::Numeric) test [metric](Metric).
    pub fn metric_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <TestOutput<EC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_test_metric_numeric(metric);
        self
    }

    /// Register a text test [metric](Metric).
    pub fn metric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + 'static,
        <TestOutput<EC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_test_metric(metric);
        self
    }

    /// Replace the default CLI renderer with a custom one.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The custom renderer.
    pub fn renderer(mut self, renderer: Box<dyn MetricsRenderer + 'static>) -> Self {
        self.renderer = Some(renderer);
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
    #[allow(clippy::type_complexity)]
    pub fn build(mut self, model: EC::Model) -> Evaluator<EC> {
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), None));

        self.event_store
            .register_logger(FileMetricLogger::new_eval(self.directory));
        let event_store = Arc::new(EventStoreClient::new(self.event_store));

        let event_processor = AsyncProcessorEvaluation::new(FullEventProcessorEvaluation::new(
            self.metrics,
            renderer,
            event_store,
        ));

        Evaluator {
            model,
            interrupter: self.interrupter,
            event_processor,
        }
    }
}

/// Trait to fake variadic generics.
pub trait EvalMetricRegistration<EC: EvaluatorComponentTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: EvaluatorBuilder<EC>) -> EvaluatorBuilder<EC>;
}

/// Trait to fake variadic generics.
pub trait EvalTextMetricRegistration<EC: EvaluatorComponentTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: EvaluatorBuilder<EC>) -> EvaluatorBuilder<EC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* EC: EvaluatorComponentTypes> EvalTextMetricRegistration<EC> for ($($M,)*)
        where
            $(<TestOutput<EC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC>,
            ) -> EvaluatorBuilder<EC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric($M);)*
                builder
            }
        }

        impl<$($M,)* EC: EvaluatorComponentTypes> EvalMetricRegistration<EC> for ($($M,)*)
        where
            $(<TestOutput<EC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + $crate::metric::Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: EvaluatorBuilder<EC>,
            ) -> EvaluatorBuilder<EC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_numeric($M);)*
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
