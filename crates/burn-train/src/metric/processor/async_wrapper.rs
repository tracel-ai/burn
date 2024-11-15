use super::{Event, EventProcessor};
use async_channel::{Receiver, Sender};

pub struct AsyncProcessor<P: EventProcessor> {
    sender: Sender<Message<P>>,
}

struct Worker<P: EventProcessor> {
    processor: P,
    rec: Receiver<Message<P>>,
}

impl<P: EventProcessor + 'static> Worker<P> {
    pub fn start(processor: P, rec: Receiver<Message<P>>) {
        let mut worker = Self { processor, rec };

        std::thread::spawn(move || {
            while let Ok(msg) = worker.rec.recv_blocking() {
                match msg {
                    Message::Train(event) => worker.processor.process_train(event),
                    Message::Valid(event) => worker.processor.process_valid(event),
                }
            }
        });
    }
}

impl<P: EventProcessor + 'static> AsyncProcessor<P> {
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        Worker::start(processor, rec);

        Self { sender }
    }
}

enum Message<P: EventProcessor> {
    Train(Event<P::ItemTrain>),
    Valid(Event<P::ItemValid>),
}

impl<P: EventProcessor> EventProcessor for AsyncProcessor<P> {
    type ItemTrain = P::ItemTrain;
    type ItemValid = P::ItemValid;

    fn process_train(&mut self, event: Event<Self::ItemTrain>) {
        self.sender.send_blocking(Message::Train(event)).unwrap();
    }

    fn process_valid(&mut self, event: Event<Self::ItemValid>) {
        self.sender.send_blocking(Message::Valid(event)).unwrap();
    }
}
