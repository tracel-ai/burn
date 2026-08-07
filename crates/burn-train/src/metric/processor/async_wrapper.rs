use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation};

use super::EventProcessorTraining;
use async_channel::{Receiver, Sender};

/// Event processor for the training process.
pub struct AsyncProcessorTraining<ET, EV> {
    sender: Sender<Message<ET, EV>>,
}

/// Event processor for the model evaluation.
pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
    sender: Sender<EvalMessage<P>>,
}

struct WorkerTraining<ET, EV, P: EventProcessorTraining<ET, EV>> {
    processor: P,
    rec: Receiver<Message<ET, EV>>,
}

struct WorkerEvaluation<P: EventProcessorEvaluation> {
    processor: P,
    rec: Receiver<EvalMessage<P>>,
}

impl<ET: Send + 'static, EV: Send + 'static, P: EventProcessorTraining<ET, EV> + 'static>
    WorkerTraining<ET, EV, P>
{
    pub fn start(processor: P, rec: Receiver<Message<ET, EV>>) {
        let mut worker = Self { processor, rec };
        std::thread::Builder::new()
            .name("train-worker".into())
            .spawn(move || {
                while let Ok(msg) = worker.rec.recv_blocking() {
                    match msg {
                        Message::Train(event) => worker.processor.process_train(event),
                        Message::Valid(event) => worker.processor.process_valid(event),
                        Message::Flush(callback) => {
                            callback.send_blocking(()).unwrap();
                        }
                        Message::Renderer(callback) => {
                            callback.send_blocking(worker.processor.renderer()).unwrap();
                            return;
                        }
                    }
                }
            })
            .unwrap();
    }
}
impl<P: EventProcessorEvaluation + 'static> WorkerEvaluation<P> {
    pub fn start(processor: P, rec: Receiver<EvalMessage<P>>) {
        let mut worker = Self { processor, rec };

        std::thread::Builder::new()
            .name("evel-worker".into())
            .spawn(move || {
                while let Ok(event) = worker.rec.recv_blocking() {
                    match event {
                        EvalMessage::Test(event) => worker.processor.process_test(event),
                        EvalMessage::Renderer(sender) => {
                            sender.send_blocking(worker.processor.renderer()).unwrap();
                            return;
                        }
                    }
                }
            })
            .unwrap();
    }
}

impl<ET: Send + 'static, EV: Send + 'static> AsyncProcessorTraining<ET, EV> {
    /// Create an event processor for training.
    pub fn new<P: EventProcessorTraining<ET, EV> + 'static>(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        WorkerTraining::start(processor, rec);

        Self { sender }
    }
}

impl<P: EventProcessorEvaluation + 'static> AsyncProcessorEvaluation<P> {
    /// Create an event processor for model evaluation.
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        WorkerEvaluation::start(processor, rec);

        Self { sender }
    }
}

enum Message<EventTrain, EventValid> {
    Train(EventTrain),
    Valid(EventValid),
    Flush(Sender<()>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

enum EvalMessage<P: EventProcessorEvaluation> {
    Test(EvaluatorEvent<P::ItemTest>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

impl<ET: Send, EV: Send> EventProcessorTraining<ET, EV> for AsyncProcessorTraining<ET, EV> {
    fn process_train(&mut self, event: ET) {
        self.sender.send_blocking(Message::Train(event)).unwrap();
    }

    fn process_valid(&mut self, event: EV) {
        self.sender.send_blocking(Message::Valid(event)).unwrap();
    }

    fn flush(&mut self) {
        let (sender, receiver) = async_channel::bounded(1);
        self.sender.send_blocking(Message::Flush(sender)).unwrap();
        receiver
            .recv_blocking()
            .expect("training event processor flush should complete");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{MetricsRenderer, cli::CliMetricsRenderer};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct TestProcessor {
        processed: Arc<AtomicUsize>,
    }

    impl EventProcessorTraining<usize, usize> for TestProcessor {
        fn process_train(&mut self, event: usize) {
            self.processed.fetch_add(event, Ordering::SeqCst);
        }

        fn process_valid(&mut self, event: usize) {
            self.processed.fetch_add(event, Ordering::SeqCst);
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn flush_waits_for_queued_training_events() {
        let processed = Arc::new(AtomicUsize::new(0));
        let mut processor = AsyncProcessorTraining::new(TestProcessor {
            processed: processed.clone(),
        });

        processor.process_train(2);
        processor.process_valid(3);
        processor.flush();

        assert_eq!(processed.load(Ordering::SeqCst), 5);
    }
}
