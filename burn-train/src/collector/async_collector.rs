use super::EventCollector;
use crate::{Aggregate, Direction, Event, Split};
use std::{sync::mpsc, thread::JoinHandle};

enum Message<T, V> {
    OnEventTrain(Event<T>),
    OnEventValid(Event<V>),
    End,
    FindEpoch(
        String,
        Aggregate,
        Direction,
        Split,
        mpsc::SyncSender<Option<usize>>,
    ),
}

/// Async [event collector](EventCollector).
///
/// This will create a worker thread where all the computation is done ensuring that the training loop is
/// never blocked by metric calculation.
pub struct AsyncEventCollector<T, V> {
    sender: mpsc::Sender<Message<T, V>>,
    handler: Option<JoinHandle<()>>,
}

#[derive(new)]
struct WorkerThread<C, T, V> {
    collector: C,
    receiver: mpsc::Receiver<Message<T, V>>,
}

impl<C, T, V> WorkerThread<C, T, V>
where
    C: EventCollector<ItemTrain = T, ItemValid = V>,
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
                Message::OnEventTrain(event) => self.collector.on_event_train(event),
                Message::OnEventValid(event) => self.collector.on_event_valid(event),
            }
        }
    }
}

impl<T: Send + Sync + 'static, V: Send + Sync + 'static> AsyncEventCollector<T, V> {
    /// Create a new async [event collector](EventCollector).
    pub fn new<C>(collector: C) -> Self
    where
        C: EventCollector<ItemTrain = T, ItemValid = V> + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let thread = WorkerThread::new(collector, receiver);

        let handler = std::thread::spawn(move || thread.run());
        let handler = Some(handler);

        Self { sender, handler }
    }
}

impl<T: Send, V: Send> EventCollector for AsyncEventCollector<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn on_event_train(&mut self, event: Event<Self::ItemTrain>) {
        self.sender.send(Message::OnEventTrain(event)).unwrap();
    }

    fn on_event_valid(&mut self, event: Event<Self::ItemValid>) {
        self.sender.send(Message::OnEventValid(event)).unwrap();
    }

    fn find_epoch(
        &mut self,
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
}

impl<T, V> Drop for AsyncEventCollector<T, V> {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
