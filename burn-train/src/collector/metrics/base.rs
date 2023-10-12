use crate::{
    info::MetricsInfo,
    metric::MetricMetadata,
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
    Aggregate, Direction, Event, EventCollector, LearnerItem, Split,
};

/// Collect training events in order to display metrics with a metrics renderer.
#[derive(new)]
pub(crate) struct RenderedMetricsEventCollector<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    renderer: Box<dyn MetricsRenderer>,
    info: MetricsInfo<T, V>,
}

impl<T, V> EventCollector for RenderedMetricsEventCollector<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    type ItemTrain = T;
    type ItemValid = V;

    fn on_event_train(&mut self, event: Event<Self::ItemTrain>) {
        match event {
            Event::ProcessedItem(item) => self.on_train_item(item),
            Event::EndEpoch(epoch) => self.on_train_end_epoch(epoch),
        }
    }

    fn on_event_valid(&mut self, event: Event<Self::ItemValid>) {
        match event {
            Event::ProcessedItem(item) => self.on_valid_item(item),
            Event::EndEpoch(epoch) => self.on_valid_end_epoch(epoch),
        }
    }

    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize> {
        self.info.find_epoch(name, aggregate, direction, split)
    }
}

impl<T, V> RenderedMetricsEventCollector<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn on_train_item(&mut self, item: LearnerItem<T>) {
        let progress = (&item).into();
        let metadata = (&item).into();

        let update = self.info.update_train(&item, &metadata);

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

        let update = self.info.update_valid(&item, &metadata);

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
        self.info.end_epoch_train(epoch);
    }

    fn on_valid_end_epoch(&mut self, epoch: usize) {
        self.info.end_epoch_valid(epoch);
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
