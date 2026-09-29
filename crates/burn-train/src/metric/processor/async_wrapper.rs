use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation};

use super::EventProcessorTraining;
use async_channel::{Receiver, Sender};
use burn_core::tensor::TensorReadError;
use std::thread::JoinHandle;

/// Event processor for the training process.
///
/// Events are processed on a separate thread. An error an event causes is only known after
/// the call that submitted it has returned and is reported by a later call:
/// [`process_train`](EventProcessorTraining::process_train),
/// [`process_valid`](EventProcessorTraining::process_valid) or
/// [`flush`](EventProcessorTraining::flush), whichever comes first.
pub struct AsyncProcessorTraining<ET, EV> {
    sender: Sender<Message<ET, EV>>,
    worker: Worker,
}

/// Event processor for the model evaluation.
///
/// Like [`AsyncProcessorTraining`], an event's error is reported by a later call.
pub struct AsyncProcessorEvaluation<P: EventProcessorEvaluation> {
    sender: Sender<EvalMessage<P>>,
    worker: Worker,
}

/// The thread a processor runs on, kept so its failure can be reported.
///
/// A processor reads tensors (metrics) where a failed computation surfaces. Those failures come
/// back as errors through `error_rec`.
struct Worker {
    handle: Option<JoinHandle<()>>,
    error_rec: Receiver<TensorReadError>,
}

impl Worker {
    fn new(handle: JoinHandle<()>, error_rec: Receiver<TensorReadError>) -> Self {
        Self {
            handle: Some(handle),
            error_rec,
        }
    }

    /// The first error the worker reported since the last call, if any.
    ///
    /// Later ones are logged and dropped: they usually share the first one's cause, and one
    /// error per call is what the caller can act on.
    fn reported(&self) -> Result<(), TensorReadError> {
        let Ok(first) = self.error_rec.try_recv() else {
            return Ok(());
        };
        while let Ok(other) = self.error_rec.try_recv() {
            log::error!("an event processor error was dropped in favor of an earlier one: {other}");
        }
        Err(first)
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
    error_sender: Sender<TensorReadError>,
}

struct WorkerEvaluation<P: EventProcessorEvaluation> {
    processor: P,
    rec: Receiver<EvalMessage<P>>,
    error_sender: Sender<TensorReadError>,
}

fn report(error_sender: &Sender<TensorReadError>, result: Result<(), TensorReadError>) {
    if let Err(err) = result {
        let _ = error_sender.try_send(err);
    }
}

impl<ET: Send + 'static, EV: Send + 'static, P: EventProcessorTraining<ET, EV> + 'static>
    WorkerTraining<ET, EV, P>
{
    pub fn start(
        processor: P,
        rec: Receiver<Message<ET, EV>>,
        error_sender: Sender<TensorReadError>,
    ) -> JoinHandle<()> {
        let mut worker = Self {
            processor,
            rec,
            error_sender,
        };
        std::thread::Builder::new()
            .name("train-worker".into())
            .spawn(move || {
                while let Ok(msg) = worker.rec.recv_blocking() {
                    match msg {
                        Message::Train(event) => {
                            report(&worker.error_sender, worker.processor.process_train(event))
                        }
                        Message::Valid(event) => {
                            report(&worker.error_sender, worker.processor.process_valid(event))
                        }
                        Message::Flush(callback) => {
                            report(&worker.error_sender, worker.processor.flush());
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
    pub fn start(
        processor: P,
        rec: Receiver<EvalMessage<P>>,
        error_sender: Sender<TensorReadError>,
    ) -> JoinHandle<()> {
        let mut worker = Self {
            processor,
            rec,
            error_sender,
        };

        std::thread::Builder::new()
            .name("evel-worker".into())
            .spawn(move || {
                while let Ok(event) = worker.rec.recv_blocking() {
                    match event {
                        EvalMessage::Test(event) => {
                            report(&worker.error_sender, worker.processor.process_test(event))
                        }
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
        let (error_sender, error_rec) = async_channel::unbounded();

        let worker = Worker::new(
            WorkerTraining::start(processor, rec, error_sender),
            error_rec,
        );

        Self { sender, worker }
    }
}

impl<P: EventProcessorEvaluation + 'static> AsyncProcessorEvaluation<P> {
    /// Create an event processor for model evaluation.
    pub fn new(processor: P) -> Self {
        let (sender, rec) = async_channel::bounded(1);
        let (error_sender, error_rec) = async_channel::unbounded();

        let worker = Worker::new(
            WorkerEvaluation::start(processor, rec, error_sender),
            error_rec,
        );

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
    fn process_train(&mut self, event: ET) -> Result<(), TensorReadError> {
        if self.sender.send_blocking(Message::Train(event)).is_err() {
            self.worker.died();
        }
        self.worker.reported()
    }

    fn process_valid(&mut self, event: EV) -> Result<(), TensorReadError> {
        if self.sender.send_blocking(Message::Valid(event)).is_err() {
            self.worker.died();
        }
        self.worker.reported()
    }

    fn flush(&mut self) -> Result<(), TensorReadError> {
        let (sender, receiver) = async_channel::bounded(1);
        if self.sender.send_blocking(Message::Flush(sender)).is_err()
            || receiver.recv_blocking().is_err()
        {
            self.worker.died();
        }
        self.worker.reported()
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

    fn process_test(
        &mut self,
        event: EvaluatorEvent<Self::ItemTest>,
    ) -> Result<(), TensorReadError> {
        if self.sender.send_blocking(EvalMessage::Test(event)).is_err() {
            self.worker.died();
        }
        self.worker.reported()
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
    use burn_std::ExecutionError;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct TestProcessor {
        processed: Arc<AtomicUsize>,
        processed_on_flush: Arc<AtomicUsize>,
    }

    impl EventProcessorTraining<usize, usize> for TestProcessor {
        fn process_train(&mut self, event: usize) -> Result<(), TensorReadError> {
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn process_valid(&mut self, event: usize) -> Result<(), TensorReadError> {
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn flush(&mut self) -> Result<(), TensorReadError> {
            self.processed_on_flush
                .store(self.processed.load(Ordering::SeqCst), Ordering::SeqCst);
            Ok(())
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

        processor.process_train(2).unwrap();
        processor.process_valid(3).unwrap();
        processor.flush().unwrap();

        assert_eq!(processed.load(Ordering::SeqCst), 5);
        // The wrapped processor is flushed before the acknowledgement is sent back.
        assert_eq!(processed_on_flush.load(Ordering::SeqCst), 5);
    }

    /// A processor whose metrics cannot read event `0`, the way they fail when the computation
    /// they read failed. Every other event is counted.
    struct FailingProcessor {
        processed: Arc<AtomicUsize>,
    }

    impl EventProcessorTraining<usize, usize> for FailingProcessor {
        fn process_train(&mut self, event: usize) -> Result<(), TensorReadError> {
            if event == 0 {
                return Err(ExecutionError::device_poisoned("the loss could not be read").into());
            }
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn process_valid(&mut self, _event: usize) -> Result<(), TensorReadError> {
            Ok(())
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn an_event_error_reaches_the_caller_and_the_worker_keeps_going() {
        let processed = Arc::new(AtomicUsize::new(0));
        let mut processor = AsyncProcessorTraining::new(FailingProcessor {
            processed: processed.clone(),
        });

        // The failing event's own call usually returns before the worker processed it, so its
        // error is reported by the flush at the latest.
        let error = processor
            .process_train(0)
            .err()
            .or_else(|| processor.flush().err())
            .expect("the event's error must reach the caller");
        assert!(
            matches!(&error, TensorReadError::Execution(err) if err.is_device_poisoned()),
            "the caller must see the metric's own error, got: {error}"
        );

        // The error was reported once, and the worker still processes what comes next.
        processor.process_train(2).unwrap();
        processor.flush().unwrap();
        assert_eq!(processed.load(Ordering::SeqCst), 2);
    }

    /// A processor that panics, the one failure that still stops the worker.
    struct PanickingProcessor;

    impl EventProcessorTraining<usize, usize> for PanickingProcessor {
        fn process_train(&mut self, _event: usize) -> Result<(), TensorReadError> {
            panic!("the processor panicked");
        }

        fn process_valid(&mut self, _event: usize) -> Result<(), TensorReadError> {
            Ok(())
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn a_worker_panic_reaches_the_caller_with_its_cause() {
        let mut processor = AsyncProcessorTraining::new(PanickingProcessor);

        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // The worker dies on the first event; the channel has room for
            // one more, so the failure is noticed by the second send or the
            // flush at the latest.
            let _ = processor.process_train(1);
            let _ = processor.process_train(2);
            let _ = processor.flush();
        }))
        .expect_err("the worker's panic must reach the caller");

        let message = payload
            .downcast_ref::<&str>()
            .map(|message| message.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(
            message.contains("the processor panicked"),
            "the caller must see the worker's own panic, got: {message:?}"
        );
    }
}
