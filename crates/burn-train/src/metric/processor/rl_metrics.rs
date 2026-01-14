use std::collections::HashMap;

use crate::{
    EpisodeSummary, ItemLazy, LearnerItem, MetricUpdater, MetricWrapper, NumericMetricUpdater,
    metric::{
        Adaptor, Metric, MetricDefinition, MetricId, MetricMetadata, Numeric, store::MetricsUpdate,
    },
};

pub(crate) struct RLMetrics<TS: ItemLazy, ES: ItemLazy> {
    train_step: Vec<Box<dyn MetricUpdater<TS::ItemSync>>>,
    env_step: Vec<Box<dyn MetricUpdater<ES::ItemSync>>>,
    env_step_valid: Vec<Box<dyn MetricUpdater<ES::ItemSync>>>,
    episode_end: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,
    episode_end_valid: Vec<Box<dyn MetricUpdater<EpisodeSummary>>>,

    train_step_numeric: Vec<Box<dyn NumericMetricUpdater<TS::ItemSync>>>,
    env_step_numeric: Vec<Box<dyn NumericMetricUpdater<ES::ItemSync>>>,
    env_step_valid_numeric: Vec<Box<dyn NumericMetricUpdater<ES::ItemSync>>>,
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
    pub(crate) fn register_env_step_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_env_step_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        ES::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_numeric.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_train_step_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        TS::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_step.push(Box::new(metric))
    }

    /// Register a training metric.
    pub(crate) fn register_train_step_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        TS::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_step_numeric.push(Box::new(metric))
    }

    /// Register a validation env-step metric.
    pub(crate) fn register_env_step_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        ES::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_valid.push(Box::new(metric))
    }

    /// Register a validation env-step numeric metric.
    pub(crate) fn register_env_step_valid_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        ES::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.env_step_valid_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric.
    pub(crate) fn register_episode_end_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end.push(Box::new(metric))
    }

    /// Register an episode-end numeric metric.
    pub(crate) fn register_episode_end_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end_numeric.push(Box::new(metric))
    }

    /// Register an episode-end metric for validation.
    pub(crate) fn register_episode_end_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.episode_end_valid.push(Box::new(metric))
    }

    /// Register an episode-end numeric metric for validation.
    pub(crate) fn register_episode_end_valid_metric_numeric<Me: Metric + Numeric + 'static>(
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
        item: &LearnerItem<TS::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.train_step.len());
        let mut entries_numeric = Vec::with_capacity(self.train_step_numeric.len());

        for metric in self.train_step.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.train_step_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the env-step metrics from an environment step item.
    pub(crate) fn update_env_step(
        &mut self,
        item: &LearnerItem<ES::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.env_step.len());
        let mut entries_numeric = Vec::with_capacity(self.env_step_numeric.len());

        for metric in self.env_step.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.env_step_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the env-step metrics for validation from an environment step item.
    pub(crate) fn update_env_step_valid(
        &mut self,
        item: &LearnerItem<ES::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.env_step_valid.len());
        let mut entries_numeric = Vec::with_capacity(self.env_step_valid_numeric.len());

        for metric in self.env_step_valid.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.env_step_valid_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the episode-end metrics from an episode summary.
    pub(crate) fn update_episode_end(
        &mut self,
        item: &LearnerItem<EpisodeSummary>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.episode_end.len());
        let mut entries_numeric = Vec::with_capacity(self.episode_end_numeric.len());

        for metric in self.episode_end.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.episode_end_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the episode-end metrics for validation from an episode summary.
    pub(crate) fn update_episode_end_valid(
        &mut self,
        item: &LearnerItem<EpisodeSummary>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.episode_end_valid.len());
        let mut entries_numeric = Vec::with_capacity(self.episode_end_valid_numeric.len());

        for metric in self.episode_end_valid.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.episode_end_valid_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }
}
