use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation};

use super::EventProcessorTraining;
use async_channel::{Receiver, Sender};
use std::thread::JoinHandle;

/// Event processor for the training process.
pub struct AsyncProcessorTraining<ET, EV> {
    sender: Sender<Message<ET, EV>>,
    worker: Worker,
}

/// Event processor for the model evaluation.
pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
    sender: Sender<EvalMessage<P>>,
    worker: Worker,
}

/// The thread a processor runs on, kept so its failure can be reported.
///
/// A processor reads tensors (metrics) where a failed computation surfaces.
struct Worker {
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    fn new(handle: JoinHandle<()>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    /// The channel to the worker is closed: re-raise whatever stopped it.
    fn died(&mut self) -> ! {
        match self.handle.take().map(JoinHandle::join) {
            Some(Err(payload)) => std::panic::resume_unwind(payload),
            _ => panic!("the event processor worker stopped without reporting an error"),
        }
    }
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
    pub fn start(processor: P, rec: Receiver<Message<ET, EV>>) -> JoinHandle<()> {
        let mut worker = Self { processor, rec };
        std::thread::Builder::new()
            .name("train-worker".into())
            .spawn(move || {
                while let Ok(msg) = worker.rec.recv_blocking() {
                    match msg {
                        Message::Train(event) => worker.processor.process_train(event),
                        Message::Valid(event) => worker.processor.process_valid(event),
                        Message::Flush(callback) => {
                            worker.processor.flush();
                            callback.send_blocking(()).unwrap();
                        }
                        Message::Renderer(callback) => {
                            callback.send_blocking(worker.processor.renderer()).unwrap();
                            return;
                        }
                    }
                }
            })
            .unwrap()
    }
}
impl<P: EventProcessorEvaluation + 'static> WorkerEvaluation<P> {
    pub fn start(processor: P, rec: Receiver<EvalMessage<P>>) -> JoinHandle<()> {
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
            .unwrap()
    }
}

impl<ET: Send + 'static, EV: Send + 'static> AsyncProcessorTraining<ET, EV> {
    /// Create an event processor for training.
    pub fn new<P: EventProcessorTraining<ET, EV> + 'static>(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        let worker = Worker::new(WorkerTraining::start(processor, rec));

        Self { sender, worker }
    }
}

impl<P: EventProcessorEvaluation + 'static> AsyncProcessorEvaluation<P> {
    /// Create an event processor for model evaluation.
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);

        let worker = Worker::new(WorkerEvaluation::start(processor, rec));

        Self { sender, worker }
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
        if self.sender.send_blocking(Message::Train(event)).is_err() {
            self.worker.died();
        }
    }

    fn process_valid(&mut self, event: EV) {
        if self.sender.send_blocking(Message::Valid(event)).is_err() {
            self.worker.died();
        }
    }

    fn flush(&mut self) {
        let (sender, receiver) = async_channel::bounded(1);
        if self.sender.send_blocking(Message::Flush(sender)).is_err()
            || receiver.recv_blocking().is_err()
        {
            self.worker.died();
        }
    }

    fn renderer(mut self) -> Box<dyn crate::renderer::MetricsRenderer> {
        let (sender, rec) = async_channel::bounded(1);
        if self
            .sender
            .send_blocking(Message::Renderer(sender))
            .is_err()
        {
            self.worker.died();
        }

        match rec.recv_blocking() {
            Ok(value) => value,
            Err(_) => self.worker.died(),
        }
    }
}

impl<P: EventProcessorEvaluation> EventProcessorEvaluation for AsyncProcessorEvaluation<P> {
    type ItemTest = P::ItemTest;

    fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
        if self.sender.send_blocking(EvalMessage::Test(event)).is_err() {
            self.worker.died();
        }
    }

    fn renderer(mut self) -> Box<dyn crate::renderer::MetricsRenderer> {
        let (sender, rec) = async_channel::bounded(1);
        if self
            .sender
            .send_blocking(EvalMessage::Renderer(sender))
            .is_err()
        {
            self.worker.died();
        }

        match rec.recv_blocking() {
            Ok(value) => value,
            Err(_) => self.worker.died(),
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
        processed_on_flush: Arc<AtomicUsize>,
    }

    impl EventProcessorTraining<usize, usize> for TestProcessor {
        fn process_train(&mut self, event: usize) {
            self.processed.fetch_add(event, Ordering::SeqCst);
        }

        fn process_valid(&mut self, event: usize) {
            self.processed.fetch_add(event, Ordering::SeqCst);
        }

        fn flush(&mut self) {
            self.processed_on_flush
                .store(self.processed.load(Ordering::SeqCst), Ordering::SeqCst);
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn flush_waits_for_queued_training_events() {
        let processed = Arc::new(AtomicUsize::new(0));
        let processed_on_flush = Arc::new(AtomicUsize::new(0));
        let mut processor = AsyncProcessorTraining::new(TestProcessor {
            processed: processed.clone(),
            processed_on_flush: processed_on_flush.clone(),
        });

        processor.process_train(2);
        processor.process_valid(3);
        processor.flush();

        assert_eq!(processed.load(Ordering::SeqCst), 5);
        // The wrapped processor is flushed before the acknowledgement is sent back.
        assert_eq!(processed_on_flush.load(Ordering::SeqCst), 5);
    }

    /// A processor that fails the way a metric does when the computation it
    /// reads failed: by panicking on the worker thread.
    struct FailingProcessor;

    impl EventProcessorTraining<usize, usize> for FailingProcessor {
        fn process_train(&mut self, _event: usize) {
            panic!("the loss could not be read: the device is poisoned");
        }

        fn process_valid(&mut self, _event: usize) {}

        fn flush(&mut self) {}

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn a_worker_failure_reaches_the_caller_with_its_cause() {
        let mut processor = AsyncProcessorTraining::new(FailingProcessor);

        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // The worker dies on the first event; the channel has room for
            // one more, so the failure is noticed by the second send or the
            // flush at the latest.
            processor.process_train(1);
            processor.process_train(2);
            processor.flush();
        }))
        .expect_err("the worker's failure must reach the caller");

        let message = payload
            .downcast_ref::<&str>()
            .map(|message| message.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(
            message.contains("the device is poisoned"),
            "the caller must see the worker's own panic, got: {message:?}"
        );
    }
}
