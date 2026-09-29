use crate::metric::processor::{EvaluatorEvent, EventProcessorError, EventProcessorEvaluation};

use super::EventProcessorTraining;
use async_channel::{Receiver, Sender};
use std::thread::JoinHandle;

/// Event processor for the training process.
///
/// Events are processed on a separate thread. An error an event causes is only known after
/// the call that submitted it has returned and is reported by a later call:
/// [`process_train`](EventProcessorTraining::process_train),
/// [`process_valid`](EventProcessorTraining::process_valid) or
/// [`flush`](EventProcessorTraining::flush), whichever comes first. That call returns every
/// failure since the last one it reported, as one [`ProcessorError`].
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
    error_rec: Receiver<EventProcessorError>,
}

impl Worker {
    fn new(handle: JoinHandle<()>, error_rec: Receiver<EventProcessorError>) -> Self {
        Self {
            handle: Some(handle),
            error_rec,
        }
    }

    /// Every error the worker reported since the last call, if any, merged in order.
    fn check_errors(&self) -> Result<(), EventProcessorError> {
        let Ok(mut reported) = self.error_rec.try_recv() else {
            return Ok(());
        };
        while let Ok(later) = self.error_rec.try_recv() {
            reported.merge(later);
        }
        Err(reported)
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
    error_sender: Sender<EventProcessorError>,
}

struct WorkerEvaluation<P: EventProcessorEvaluation> {
    processor: P,
    rec: Receiver<EvalMessage<P>>,
    error_sender: Sender<EventProcessorError>,
}

fn report(error_sender: &Sender<EventProcessorError>, result: Result<(), EventProcessorError>) {
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
        error_sender: Sender<EventProcessorError>,
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
        error_sender: Sender<EventProcessorError>,
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
                        EvalMessage::Flush(callback) => {
                            report(&worker.error_sender, worker.processor.flush());
                            callback.send_blocking(()).unwrap();
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
    Flush(Sender<()>),
    Renderer(Sender<Box<dyn crate::renderer::MetricsRenderer>>),
}

impl<ET: Send, EV: Send> EventProcessorTraining<ET, EV> for AsyncProcessorTraining<ET, EV> {
    fn process_train(&mut self, event: ET) -> Result<(), EventProcessorError> {
        if self.sender.send_blocking(Message::Train(event)).is_err() {
            self.worker.died();
        }
        self.worker.check_errors()
    }

    fn process_valid(&mut self, event: EV) -> Result<(), EventProcessorError> {
        if self.sender.send_blocking(Message::Valid(event)).is_err() {
            self.worker.died();
        }
        self.worker.check_errors()
    }

    fn flush(&mut self) -> Result<(), EventProcessorError> {
        let (sender, receiver) = async_channel::bounded(1);
        if self.sender.send_blocking(Message::Flush(sender)).is_err()
            || receiver.recv_blocking().is_err()
        {
            self.worker.died();
        }
        self.worker.check_errors()
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
    ) -> Result<(), EventProcessorError> {
        if self.sender.send_blocking(EvalMessage::Test(event)).is_err() {
            self.worker.died();
        }
        self.worker.check_errors()
    }

    fn flush(&mut self) -> Result<(), EventProcessorError> {
        let (sender, receiver) = async_channel::bounded(1);
        if self
            .sender
            .send_blocking(EvalMessage::Flush(sender))
            .is_err()
            || receiver.recv_blocking().is_err()
        {
            self.worker.died();
        }
        self.worker.check_errors()
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
    use crate::metric::{
        processor::{EventProcessorFailure, ItemLazy, MetricError},
        store::Split,
    };
    use crate::renderer::{MetricsRenderer, cli::CliMetricsRenderer};
    use burn_core::tensor::TensorReadError;
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
        fn process_train(&mut self, event: usize) -> Result<(), EventProcessorError> {
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn process_valid(&mut self, event: usize) -> Result<(), EventProcessorError> {
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn flush(&mut self) -> Result<(), EventProcessorError> {
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

    /// A processor whose metric cannot read events `0` and `1`, the way metrics fail when the
    /// computation they read failed. Every other event is counted.
    struct FailingProcessor {
        processed: Arc<AtomicUsize>,
    }

    impl EventProcessorTraining<usize, usize> for FailingProcessor {
        fn process_train(&mut self, event: usize) -> Result<(), EventProcessorError> {
            if event < 2 {
                return EventProcessorError::from_errors(vec![MetricError {
                    metric: Arc::new(format!("event-{event}")),
                    split: Split::Train,
                    source: ExecutionError::device_poisoned("the loss could not be read").into(),
                }]);
            }
            self.processed.fetch_add(event, Ordering::SeqCst);
            Ok(())
        }

        fn process_valid(&mut self, _event: usize) -> Result<(), EventProcessorError> {
            Ok(())
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn every_event_error_reaches_the_caller_and_the_worker_keeps_going() {
        let processed = Arc::new(AtomicUsize::new(0));
        let mut processor = AsyncProcessorTraining::new(FailingProcessor {
            processed: processed.clone(),
        });

        // Each call returns whatever failed since the last report, so the failures are split
        // across these calls depending on how far the worker got; the flush gets the rest.
        let reports = [
            processor.process_train(0),
            processor.process_train(1),
            processor.flush(),
        ];
        let failures: Vec<MetricError> = reports
            .into_iter()
            .filter_map(Result::err)
            .flat_map(EventProcessorError::into_failures)
            .map(|failure| match failure {
                EventProcessorFailure::Metric(error) => error,
                other => panic!("expected a metric failure, got {other}"),
            })
            .collect();

        let failed: Vec<&str> = failures.iter().map(|err| err.metric.as_str()).collect();
        assert_eq!(failed, ["event-0", "event-1"], "every failure, in order");
        assert!(failures.iter().all(|err| matches!(
            &err.source,
            TensorReadError::Execution(err) if err.is_device_poisoned()
        )));

        // Each failure was reported once, and the worker still processes what comes next.
        processor.process_train(2).unwrap();
        processor.flush().unwrap();
        assert_eq!(processed.load(Ordering::SeqCst), 2);
    }

    /// An evaluation processor whose metric fails at the end of every test split.
    struct FailingEvalProcessor;

    struct NoItem;

    impl ItemLazy for NoItem {
        fn sync(self) -> Result<Self, ExecutionError> {
            Ok(self)
        }
    }

    impl EventProcessorEvaluation for FailingEvalProcessor {
        type ItemTest = NoItem;

        fn process_test(
            &mut self,
            event: EvaluatorEvent<NoItem>,
        ) -> Result<(), EventProcessorError> {
            match event {
                EvaluatorEvent::EndTest => EventProcessorError::from_errors(vec![MetricError {
                    metric: Arc::new("Accuracy".to_string()),
                    split: Split::Test(None),
                    source: ExecutionError::with_context("refused").into(),
                }]),
                _ => Ok(()),
            }
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            Box::new(CliMetricsRenderer::new())
        }
    }

    #[test]
    fn flush_reports_last_events_errors() {
        let mut processor = AsyncProcessorEvaluation::new(FailingEvalProcessor);

        // The last event's own call may return before the worker processed it: the flush
        // is what guarantees its error is seen before the processor is consumed.
        let reported = processor
            .process_test(EvaluatorEvent::EndTest)
            .err()
            .or_else(|| processor.flush().err())
            .expect("the last event's error must reach the caller");
        match &reported.failures()[0] {
            EventProcessorFailure::Metric(error) => assert_eq!(error.metric.as_str(), "Accuracy"),
            other => panic!("expected a metric failure, got {other}"),
        }

        processor.flush().unwrap();
    }

    /// A processor that panics, the one failure that still stops the worker.
    struct PanickingProcessor;

    impl EventProcessorTraining<usize, usize> for PanickingProcessor {
        fn process_train(&mut self, _event: usize) -> Result<(), EventProcessorError> {
            panic!("the processor panicked");
        }

        fn process_valid(&mut self, _event: usize) -> Result<(), EventProcessorError> {
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
