use crate::{
    info::Metrics,
    metric::MetricMetadata,
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
    Aggregate, Direction, LearnerItem, Split, TrainingEvent, TrainingEventCollector,
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

impl<T, V> TrainingEventCollector for MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    type ItemTrain = T;
    type ItemValid = V;

    fn on_event_train(&mut self, event: TrainingEvent<Self::ItemTrain>) {
        match event {
            TrainingEvent::ProcessedItem(item) => self.on_train_item(item),
            TrainingEvent::EndEpoch(epoch) => self.on_train_end_epoch(epoch),
        }
    }

    fn on_event_valid(&mut self, event: TrainingEvent<Self::ItemValid>) {
        match event {
            TrainingEvent::ProcessedItem(item) => self.on_valid_item(item),
            TrainingEvent::EndEpoch(epoch) => self.on_valid_end_epoch(epoch),
        }
    }

    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize> {
        self.metrics.find_epoch(name, aggregate, direction, split)
    }
}

impl<T, V> MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
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
