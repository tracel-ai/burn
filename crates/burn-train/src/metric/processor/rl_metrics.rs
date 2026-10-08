use std::collections::HashMap;

use crate::{
    EpisodeSummary, EvaluationItem, ItemLazy, MetricError, MetricUpdater, MetricWrapper,
    NumericMetricUpdater,
    metric::{
        Adaptor, Metric, MetricDefinition, MetricId, MetricMetadata, Numeric,
        processor::update_metrics,
        store::{MetricsUpdate, Split},
    },
};

pub(crate) struct RLMetrics<TS: ItemLazy, ES: ItemLazy> {
    train_step: Vec<Box<dyn MetricUpdater<TS>>>,
    env_step: Vec<Box<dyn MetricUpdater<ES>>>,
    env_step_valid: Vec<Box<dyn MetricUpdater<ES>>>,
    episode_end: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,
    episode_end_valid: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,

    train_step_numeric: Vec<Box<dyn NumericMetricUpdater<TS>>>,
    env_step_numeric: Vec<Box<dyn NumericMetricUpdater<ES>>>,
    env_step_valid_numeric: Vec<Box<dyn NumericMetricUpdater<ES>>>,
    episode_end_numeric: Vec<Box<dyn NumericMetricUpdater<EpisodeSummary>>>,
    episode_end_valid_numeric: Vec<Box<dyn NumericMetricUpdater<EpisodeSummary>>>,

    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

impl<TS: ItemLazy, ES: ItemLazy> Default for RLMetrics<TS, ES> {
    fn default() -> Self {
        Self {
            train_step: Vec::default(),
            env_step: Vec::default(),
            env_step_valid: Vec::default(),
            episode_end: Vec::default(),
            episode_end_valid: Vec::default(),
            train_step_numeric: Vec::default(),
            env_step_numeric: Vec::default(),
            env_step_valid_numeric: Vec::default(),
            episode_end_numeric: Vec::default(),
            episode_end_valid_numeric: Vec::default(),
            metric_definitions: HashMap::default(),
        }
    }
}

impl<TS: ItemLazy, ES: ItemLazy> RLMetrics<TS, ES> {
    /// Register a training metric.
    pub(crate) fn register_text_metric_agent<Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_agent_metric<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_numeric.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_text_metric_train<Me: Metric + 'static>(&mut self, metric: Me)
    where
        TS: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_metric_train<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        TS: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_step_numeric.push(Box::new(metric))
    }

    /// Register a validation env-step metric.
    pub(crate) fn register_text_metric_agent_valid<Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_valid.push(Box::new(metric))
    }

    /// Register a validation env-step numeric metric.
    pub(crate) fn register_agent_metric_valid<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_valid_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric.
    pub(crate) fn register_text_metric_episode<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end.push(Box::new(metric))
    }

    /// Register an episode-end numeric metric.
    pub(crate) fn register_episode_metric<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric for validation.
    pub(crate) fn register_text_metric_episode_valid<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end_valid.push(Box::new(metric))
    }

    /// Register an episode-end numeric metric for validation.
    pub(crate) fn register_episode_metric_valid<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end_valid_numeric.push(Box::new(metric))
    }

    fn register_definition<Me: Metric>(&mut self, metric: &MetricWrapper<Me>) {
        self.metric_definitions.insert(
            metric.id.clone(),
            MetricDefinition::new(metric.id.clone(), &metric.metric),
        );
    }

    /// Get metric definitions for all splits
    pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
        self.metric_definitions.values().cloned().collect()
    }

    /// Update the training information from the training item.
    pub(crate) fn update_train_step(
        &mut self,
        item: &EvaluationItem<TS>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.train_step,
            &mut self.train_step_numeric,
            &item.item,
            metadata,
            Split::Train,
        )
    }

    /// Update the env-step metrics from an environment step item.
    pub(crate) fn update_env_step(
        &mut self,
        item: &EvaluationItem<ES>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.env_step,
            &mut self.env_step_numeric,
            &item.item,
            metadata,
            Split::Train,
        )
    }

    /// Update the env-step metrics for validation from an environment step item.
    pub(crate) fn update_env_step_valid(
        &mut self,
        item: &EvaluationItem<ES>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.env_step_valid,
            &mut self.env_step_valid_numeric,
            &item.item,
            metadata,
            Split::Valid,
        )
    }

    /// Update the episode-end metrics from an episode summary.
    pub(crate) fn update_episode_end(
        &mut self,
        item: &EvaluationItem<EpisodeSummary>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.episode_end,
            &mut self.episode_end_numeric,
            &item.item,
            metadata,
            Split::Train,
        )
    }

    /// Update the episode-end metrics for validation from an episode summary.
    pub(crate) fn update_episode_end_valid(
        &mut self,
        item: &EvaluationItem<EpisodeSummary>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.episode_end_valid,
            &mut self.episode_end_valid_numeric,
            &item.item,
            metadata,
            Split::Valid,
        )
    }
}
