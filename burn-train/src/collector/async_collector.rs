use super::TrainingEventCollector;
use crate::{Aggregate, Direction, Split, TrainingEvent};
use std::{sync::mpsc, thread::JoinHandle};

enum Message<T, V> {
    OnEventTrain(TrainingEvent<T>),
    OnEventValid(TrainingEvent<V>),
    End,
    FindEpoch(
        String,
        Aggregate,
        Direction,
        Split,
        mpsc::SyncSender<Option<usize>>,
    ),
}

/// Async trainer callback tracker.
pub struct AsyncTrainerCallback<T, V> {
    sender: mpsc::Sender<Message<T, V>>,
    handler: Option<JoinHandle<()>>,
}

#[derive(new)]
struct CallbackThread<C, T, V> {
    callback: C,
    receiver: mpsc::Receiver<Message<T, V>>,
}

impl<C, T, V> CallbackThread<C, T, V>
where
    C: TrainingEventCollector<ItemTrain = T, ItemValid = V>,
{
    fn run(mut self) {
        for item in self.receiver.iter() {
            match item {
                Message::End => {
                    return;
                }
                Message::FindEpoch(name, aggregate, direction, split, sender) => {
                    let response = self.callback.find_epoch(&name, aggregate, direction, split);
                    sender.send(response).unwrap();
                }
                Message::OnEventTrain(event) => self.callback.on_event_train(event),
                Message::OnEventValid(event) => self.callback.on_event_valid(event),
            }
        }
    }
}

impl<T: Send + Sync + 'static, V: Send + Sync + 'static> AsyncTrainerCallback<T, V> {
    /// Create a new async trainer callback.
    pub fn new<C>(callback: C) -> Self
    where
        C: TrainingEventCollector<ItemTrain = T, ItemValid = V> + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let thread = CallbackThread::new(callback, receiver);

        let handler = std::thread::spawn(move || thread.run());
        let handler = Some(handler);

        Self { sender, handler }
    }
}

impl<T: Send, V: Send> TrainingEventCollector for AsyncTrainerCallback<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn on_event_train(&mut self, event: TrainingEvent<Self::ItemTrain>) {
        self.sender.send(Message::OnEventTrain(event)).unwrap();
    }

    fn on_event_valid(&mut self, event: TrainingEvent<Self::ItemValid>) {
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

impl<T, V> Drop for AsyncTrainerCallback<T, V> {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
