use crate::metric::processor::EventProcessorEvaluation;

use super::{Event, EventProcessorTraining};
use async_channel::{Receiver, Sender};

pub struct AsyncProcessorTraining<P: EventProcessorTraining> {
    sender: Sender<Message<P>>,
}

pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
    sender: Sender<Event<P::ItemTest>>,
}

struct WorkerTraining<P: EventProcessorTraining> {
    processor: P,
    rec: Receiver<Message<P>>,
}

struct WorkerEvaluation<P: EventProcessorEvaluation> {
    processor: P,
    rec: Receiver<Event<P::ItemTest>>,
}

impl<P: EventProcessorTraining + 'static> WorkerTraining<P> {
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
impl<P: EventProcessorEvaluation + 'static> WorkerEvaluation<P> {
    pub fn start(processor: P, rec: Receiver<Event<P::ItemTest>>) {
        let mut worker = Self { processor, rec };

        std::thread::spawn(move || {
            while let Ok(event) = worker.rec.recv_blocking() {
                worker.processor.process_test(event)
            }
        });
    }
}

impl<P: EventProcessorTraining + 'static> AsyncProcessorTraining<P> {
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        WorkerTraining::start(processor, rec);

        Self { sender }
    }
}

impl<P: EventProcessorEvaluation + 'static> AsyncProcessorEvaluation<P> {
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        WorkerEvaluation::start(processor, rec);

        Self { sender }
    }
}

enum Message<P: EventProcessorTraining> {
    Train(Event<P::ItemTrain>),
    Valid(Event<P::ItemValid>),
}

impl<P: EventProcessorTraining> EventProcessorTraining for AsyncProcessorTraining<P> {
    type ItemTrain = P::ItemTrain;
    type ItemValid = P::ItemValid;

    fn process_train(&mut self, event: Event<Self::ItemTrain>) {
        self.sender.send_blocking(Message::Train(event)).unwrap();
    }

    fn process_valid(&mut self, event: Event<Self::ItemValid>) {
        self.sender.send_blocking(Message::Valid(event)).unwrap();
    }
}

impl<P: EventProcessorEvaluation> EventProcessorEvaluation for AsyncProcessorEvaluation<P> {
    type ItemTest = P::ItemTest;

    fn process_test(&mut self, event: Event<Self::ItemTest>) {
        self.sender.send_blocking(event).unwrap();
    }
}
