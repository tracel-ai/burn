use super::EventStore;
use super::{Aggregate, Direction, Event, Split};
use std::{sync::mpsc, thread::JoinHandle};

/// Type that allows to communicate with an [event store](EventStore).
pub struct EventStoreClient {
    sender: mpsc::Sender<Message>,
    handler: Option<JoinHandle<()>>,
}

impl EventStoreClient {
    /// Create a new [event store](EventStore) client.
    pub(crate) fn new<C>(store: C) -> Self
    where
        C: EventStore + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let thread = WorkerThread::new(store, receiver);

        let handler = std::thread::spawn(move || thread.run());
        let handler = Some(handler);

        Self { sender, handler }
    }
}

impl EventStoreClient {
    /// Add a training event to the [event store](EventStore).
    pub(crate) fn add_event_train(&self, event: Event) {
        self.sender
            .send(Message::OnEventTrain(event))
            .expect("Can send event to event store thread.");
    }

    /// Add a validation event to the [event store](EventStore).
    pub(crate) fn add_event_valid(&self, event: Event) {
        self.sender
            .send(Message::OnEventValid(event))
            .expect("Can send event to event store thread.");
    }

    /// Add a testing event to the [event store](EventStore).
    pub(crate) fn add_event_test(&self, event: Event) {
        self.sender
            .send(Message::OnEventTest(event))
            .expect("Can send event to event store thread.");
    }

    /// Find the epoch following the given criteria from the collected data.
    pub fn find_epoch(
        &self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Message::FindEpoch(
                name.to_string(),
                aggregate,
                direction,
                split,
                sender,
            ))
            .expect("Can send event to event store thread.");

        match receiver.recv() {
            Ok(value) => value,
            Err(err) => panic!("Event store thread crashed: {err:?}"),
        }
    }

    /// Find the metric value for the current epoch following the given criteria.
    pub fn find_metric(
        &self,
        name: &str,
        epoch: usize,
        aggregate: Aggregate,
        split: Split,
    ) -> Option<f64> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Message::FindMetric(
                name.to_string(),
                epoch,
                aggregate,
                split,
                sender,
            ))
            .expect("Can send event to event store thread.");

        match receiver.recv() {
            Ok(value) => value,
            Err(err) => panic!("Event store thread crashed: {err:?}"),
        }
    }
}

#[derive(new)]
struct WorkerThread<S> {
    store: S,
    receiver: mpsc::Receiver<Message>,
}

impl<C> WorkerThread<C>
where
    C: EventStore,
{
    fn run(mut self) {
        for item in self.receiver.iter() {
            match item {
                Message::End => {
                    return;
                }
                Message::FindEpoch(name, aggregate, direction, split, callback) => {
                    let response = self.store.find_epoch(&name, aggregate, direction, split);
                    callback
                        .send(response)
                        .expect("Can send response using callback channel.");
                }
                Message::FindMetric(name, epoch, aggregate, split, callback) => {
                    let response = self.store.find_metric(&name, epoch, aggregate, split);
                    callback
                        .send(response)
                        .expect("Can send response using callback channel.");
                }
                Message::OnEventTrain(event) => self.store.add_event(event, Split::Train),
                Message::OnEventValid(event) => self.store.add_event(event, Split::Valid),
                Message::OnEventTest(event) => self.store.add_event(event, Split::Test),
            }
        }
    }
}

enum Message {
    OnEventTest(Event),
    OnEventTrain(Event),
    OnEventValid(Event),
    End,
    FindEpoch(
        String,
        Aggregate,
        Direction,
        Split,
        mpsc::SyncSender<Option<usize>>,
    ),
    FindMetric(
        String,
        usize,
        Aggregate,
        Split,
        mpsc::SyncSender<Option<f64>>,
    ),
}

impl Drop for EventStoreClient {
    fn drop(&mut self) {
        self.sender
            .send(Message::End)
            .expect("Can send the end message to the event store thread.");
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().expect("The event store thread should stop.");
        }
    }
}
