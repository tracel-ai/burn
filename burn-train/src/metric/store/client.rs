use super::EventStore;
use super::{Aggregate, Direction, Event, Split};
use std::{sync::mpsc, thread::JoinHandle};

enum Message {
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

/// Async [event collector](EventCollector).
///
/// This will create a worker thread where all the computation is done ensuring that the training loop is
/// never blocked by metric calculation.
pub struct EventStoreClient {
    sender: mpsc::Sender<Message>,
    handler: Option<JoinHandle<()>>,
}

impl EventStoreClient {
    /// Create a new async [event collector](EventCollector).
    pub fn new<C>(store: C) -> Self
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
    pub fn add_event_train(&self, event: Event) {
        self.sender.send(Message::OnEventTrain(event)).unwrap();
    }

    pub fn add_event_valid(&self, event: Event) {
        self.sender.send(Message::OnEventValid(event)).unwrap();
    }

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
            .unwrap();

        match receiver.recv() {
            Ok(value) => value,
            Err(err) => panic!("Async server crashed: {:?}", err),
        }
    }

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
            .unwrap();

        match receiver.recv() {
            Ok(value) => value,
            Err(err) => panic!("Async server crashed: {:?}", err),
        }
    }
}

#[derive(new)]
struct WorkerThread<C> {
    collector: C,
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
                Message::FindEpoch(name, aggregate, direction, split, sender) => {
                    let response = self
                        .collector
                        .find_epoch(&name, aggregate, direction, split);
                    sender.send(response).unwrap();
                }
                Message::FindMetric(name, epoch, aggregate, split, sender) => {
                    let response = self.collector.find_metric(&name, epoch, aggregate, split);
                    sender.send(response).unwrap();
                }
                Message::OnEventTrain(event) => self.collector.add_event(event, Split::Train),
                Message::OnEventValid(event) => self.collector.add_event(event, Split::Valid),
            }
        }
    }
}

impl Drop for EventStoreClient {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
