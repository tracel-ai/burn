use std::collections::HashMap;

use crate::{
    EpisodeSummary, EvaluationItem, MetricUpdater, MetricWrapper, NumericMetricUpdater,
    metric::{
        Adaptor, Metric, MetricDefinition, MetricId, MetricMetadata, Numeric, store::MetricsUpdate,
    },
};

#[derive(Default)]
pub(crate) struct RLMetrics {
    train_step: Vec<Box<dyn MetricUpdater>>,
    env_step: Vec<Box<dyn MetricUpdater>>,
    env_step_valid: Vec<Box<dyn MetricUpdater>>,
    episode_end: Vec<Box<dyn MetricUpdater>>,
    episode_end_valid: Vec<Box<dyn MetricUpdater>>,

    train_step_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    env_step_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    env_step_valid_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    episode_end_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    episode_end_valid_numeric: Vec<Box<dyn NumericMetricUpdater>>,

    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

impl RLMetrics {
    /// Register a training metric.
    pub(crate) fn register_text_metric_agent<ES, Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<ES, _>::new(metric);
        self.register_definition(&metric);
        self.env_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_agent_metric<ES, Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<ES, _>::new(metric);
        self.register_definition(&metric);
        self.env_step_numeric.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_text_metric_train<TS, Me: Metric + 'static>(&mut self, metric: Me)
    where
        TS: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<TS, _>::new(metric);
        self.register_definition(&metric);
        self.train_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_metric_train<TS, Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        TS: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<TS, _>::new(metric);
        self.register_definition(&metric);
        self.train_step_numeric.push(Box::new(metric))
    }

    /// Register a validation env-step metric.
    pub(crate) fn register_text_metric_agent_valid<ES, Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<ES, _>::new(metric);
        self.register_definition(&metric);
        self.env_step_valid.push(Box::new(metric))
    }

    /// Register a validation env-step numeric metric.
    pub(crate) fn register_agent_metric_valid<ES, Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        ES: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<ES, _>::new(metric);
        self.register_definition(&metric);
        self.env_step_valid_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric.
    pub(crate) fn register_text_metric_episode<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<EpisodeSummary, _>::new(metric);
        self.register_definition(&metric);
        self.episode_end.push(Box::new(metric))
    }

    /// Register an episode-end numeric metric.
    pub(crate) fn register_episode_metric<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<EpisodeSummary, _>::new(metric);
        self.register_definition(&metric);
        self.episode_end_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric for validation.
    pub(crate) fn register_text_metric_episode_valid<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<EpisodeSummary, _>::new(metric);
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
        let metric = MetricWrapper::<EpisodeSummary, _>::new(metric);
        self.register_definition(&metric);
        self.episode_end_valid_numeric.push(Box::new(metric))
    }

    fn register_definition<T, Me: Metric>(&mut self, metric: &MetricWrapper<T, Me>) {
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
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.train_step.len());
        let mut entries_numeric = Vec::with_capacity(self.train_step_numeric.len());

        for metric in self.train_step.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.train_step_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the env-step metrics from an environment step item.
    pub(crate) fn update_env_step(
        &mut self,
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.env_step.len());
        let mut entries_numeric = Vec::with_capacity(self.env_step_numeric.len());

        for metric in self.env_step.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.env_step_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the env-step metrics for validation from an environment step item.
    pub(crate) fn update_env_step_valid(
        &mut self,
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.env_step_valid.len());
        let mut entries_numeric = Vec::with_capacity(self.env_step_valid_numeric.len());

        for metric in self.env_step_valid.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.env_step_valid_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the episode-end metrics from an episode summary.
    pub(crate) fn update_episode_end(
        &mut self,
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.episode_end.len());
        let mut entries_numeric = Vec::with_capacity(self.episode_end_numeric.len());

        for metric in self.episode_end.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.episode_end_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the episode-end metrics for validation from an episode summary.
    pub(crate) fn update_episode_end_valid(
        &mut self,
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.episode_end_valid.len());
        let mut entries_numeric = Vec::with_capacity(self.episode_end_valid_numeric.len());

        for metric in self.episode_end_valid.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.episode_end_valid_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }
}
