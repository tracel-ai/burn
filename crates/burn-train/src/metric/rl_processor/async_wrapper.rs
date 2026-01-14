use crate::metric::{
    processor::{EvaluatorEvent, EventProcessorEvaluation},
    rl_processor::{RlEvaluationEvent, RlEventProcessorTrain, RlTrainingEvent},
};

use async_channel::{Receiver, Sender};

/// Event processor for the training process.
pub struct RlAsyncProcessorTraining<P: RlEventProcessorTrain> {
    sender: Sender<Message<P>>,
}

struct WorkerTraining<P: RlEventProcessorTrain> {
    processor: P,
    rec: Receiver<Message<P>>,
}

impl<P: RlEventProcessorTrain + 'static> WorkerTraining<P> {
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

impl<P: RlEventProcessorTrain + 'static> RlAsyncProcessorTraining<P> {
    /// Create an event processor for training.
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        WorkerTraining::start(processor, rec);

        Self { sender }
    }
}

enum Message<P: RlEventProcessorTrain> {
    Train(RlTrainingEvent<P::TrainingOutput, P::ActionContext>),
    Valid(RlEvaluationEvent<P::ActionContext>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

enum EvalMessage<P: EventProcessorEvaluation> {
    Test(EvaluatorEvent<P::ItemTest>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

impl<P: RlEventProcessorTrain> RlEventProcessorTrain for RlAsyncProcessorTraining<P> {
    type TrainingOutput = P::TrainingOutput;
    type ActionContext = P::ActionContext;

    fn process_train(&mut self, event: RlTrainingEvent<P::TrainingOutput, P::ActionContext>) {
        self.sender.send_blocking(Message::Train(event)).unwrap();
    }

    fn process_valid(&mut self, event: RlEvaluationEvent<P::ActionContext>) {
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
