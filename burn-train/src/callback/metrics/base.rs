use crate::{
    info::Metrics,
    metric::MetricMetadata,
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
    LearnerCallback, LearnerItem,
};

/// Holds all metrics, metric loggers, and a metrics renderer.
#[derive(new)]
pub(crate) struct MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    renderer: Box<dyn MetricsRenderer>,
    metrics: Metrics<T, V>,
}

impl<T, V> LearnerCallback for MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    type ItemTrain = T;
    type ItemValid = V;

    fn on_train_item(&mut self, item: LearnerItem<T>) {
        let progress = (&item).into();
        let metadata = (&item).into();

        let update = self.metrics.update_train(&item, &metadata);

        update
            .entries
            .into_iter()
            .for_each(|entry| self.renderer.update_train(MetricState::Generic(entry)));

        update
            .entries_numeric
            .into_iter()
            .for_each(|(entry, value)| {
                self.renderer
                    .update_train(MetricState::Numeric(entry, value))
            });

        self.renderer.render_train(progress);
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        let progress = (&item).into();
        let metadata = (&item).into();

        let update = self.metrics.update_valid(&item, &metadata);

        update
            .entries
            .into_iter()
            .for_each(|entry| self.renderer.update_valid(MetricState::Generic(entry)));

        update
            .entries_numeric
            .into_iter()
            .for_each(|(entry, value)| {
                self.renderer
                    .update_valid(MetricState::Numeric(entry, value))
            });

        self.renderer.render_train(progress);
    }

    fn on_train_end_epoch(&mut self, epoch: usize) {
        self.metrics.end_epoch_train(epoch);
    }

    fn on_valid_end_epoch(&mut self, epoch: usize) {
        self.metrics.end_epoch_valid(epoch);
    }
}

impl<T> From<&LearnerItem<T>> for TrainingProgress {
    fn from(item: &LearnerItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            epoch: item.epoch,
            epoch_total: item.epoch_total,
            iteration: item.iteration,
        }
    }
}

impl<T> From<&LearnerItem<T>> for MetricMetadata {
    fn from(item: &LearnerItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            epoch: item.epoch,
            epoch_total: item.epoch_total,
            iteration: item.iteration,
            lr: item.lr,
        }
    }
}
