use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation};

use super::{EventProcessorTraining, LearnerEvent};
use async_channel::{Receiver, Sender};

pub struct AsyncProcessorTraining<P: EventProcessorTraining> {
    sender: Sender<Message<P>>,
}

pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
    sender: Sender<EvalMessage<P>>,
}

struct WorkerTraining<P: EventProcessorTraining> {
    processor: P,
    rec: Receiver<Message<P>>,
}

struct WorkerEvaluation<P: EventProcessorEvaluation> {
    processor: P,
    rec: Receiver<EvalMessage<P>>,
}

impl<P: EventProcessorTraining + 'static> WorkerTraining<P> {
    pub fn start(processor: P, rec: Receiver<Message<P>>) {
        let mut worker = Self { processor, rec };

        std::thread::spawn(move || {
            while let Ok(msg) = worker.rec.recv_blocking() {
                match msg {
                    Message::Train(event) => worker.processor.process_train(event),
                    Message::Valid(event) => worker.processor.process_valid(event),
                    Message::Renderer(callback) => {
                        callback.send_blocking(worker.processor.renderer()).unwrap();
                        return;
                    }
                }
            }
        });
    }
}
impl<P: EventProcessorEvaluation + 'static> WorkerEvaluation<P> {
    pub fn start(processor: P, rec: Receiver<EvalMessage<P>>) {
        let mut worker = Self { processor, rec };

        std::thread::spawn(move || {
            while let Ok(event) = worker.rec.recv_blocking() {
                match event {
                    EvalMessage::Test(event) => worker.processor.process_test(event),
                    EvalMessage::Renderer(sender) => {
                        sender.send_blocking(worker.processor.renderer()).unwrap();
                        return;
                    }
                }
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
    Train(LearnerEvent<P::ItemTrain>),
    Valid(LearnerEvent<P::ItemValid>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

enum EvalMessage<P: EventProcessorEvaluation> {
    Test(EvaluatorEvent<P::ItemTest>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

impl<P: EventProcessorTraining> EventProcessorTraining for AsyncProcessorTraining<P> {
    type ItemTrain = P::ItemTrain;
    type ItemValid = P::ItemValid;

    fn process_train(&mut self, event: LearnerEvent<Self::ItemTrain>) {
        self.sender.send_blocking(Message::Train(event)).unwrap();
    }

    fn process_valid(&mut self, event: LearnerEvent<Self::ItemValid>) {
        self.sender.send_blocking(Message::Valid(event)).unwrap();
    }

    fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
        let (sender, rec) = async_channel::bounded(1);
        self.sender
            .send_blocking(Message::Renderer(sender))
            .unwrap();

        match rec.recv_blocking() {
            Ok(value) => value,
            Err(err) => panic!("{err:?}"),
        }
    }
}

impl<P: EventProcessorEvaluation> EventProcessorEvaluation for AsyncProcessorEvaluation<P> {
    type ItemTest = P::ItemTest;

    fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
        self.sender.send_blocking(EvalMessage::Test(event)).unwrap();
    }

    fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
        let (sender, rec) = async_channel::bounded(1);
        self.sender
            .send_blocking(EvalMessage::Renderer(sender))
            .unwrap();

        match rec.recv_blocking() {
            Ok(value) => value,
            Err(err) => panic!("{err:?}"),
        }
    }
}
