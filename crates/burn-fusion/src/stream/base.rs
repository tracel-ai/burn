use super::Operation;
use super::OperationConverter;
use super::OperationDescription;
use crate::FusionBackend;

/// A growing list of [tensor operation descriptions](OperationDescription).
pub struct OperationQueue<B: FusionBackend> {
    pub(crate) global: Vec<OperationDescription>,
    pub(crate) relative: Vec<OperationDescription>,
    pub(crate) converter: OperationConverter,
    pub(crate) operations: Vec<Box<dyn Operation<B>>>,
}

impl<B: FusionBackend> Default for OperationQueue<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// The stream id.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct StreamId {
    #[cfg(feature = "std")]
    value: std::thread::ThreadId,
    #[cfg(not(feature = "std"))]
    value: (),
}

impl StreamId {
    /// Get the current stream id.
    pub fn current() -> Self {
        Self {
            #[cfg(feature = "std")]
            value: Self::id(),
            #[cfg(not(feature = "std"))]
            value: (),
        }
    }

    #[cfg(feature = "std")]
    fn id() -> std::thread::ThreadId {
        std::thread_local! {
            static ID: std::cell::OnceCell::<std::thread::ThreadId> = const { std::cell::OnceCell::new() };
        };

        // Getting the current thread is expensive, so we cache the value into a thread local
        // variable, which is very fast.
        ID.with(|cell| *cell.get_or_init(|| std::thread::current().id()))
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("StreamID({:?})", self.value))
    }
}

impl<B: FusionBackend> OperationQueue<B> {
    /// Create a new empty queue.
    pub fn new() -> Self {
        Self {
            global: Vec::new(),
            relative: Vec::new(),
            converter: OperationConverter::default(),
            operations: Vec::new(),
        }
    }

    /// Add a new tensor operation to the queue.
    ///
    /// The new [operation description](OperationDescription) will be converted to a local
    /// representation that can be reused when the same pattern emerge in different but similar
    /// scenario, so that the same optimization can be used.
    pub fn add(&mut self, global: OperationDescription, operation: Box<dyn Operation<B>>) {
        let relative = global.to_relative(&mut self.converter);
        self.relative.push(relative);
        self.global.push(global);
        self.operations.push(operation);
    }

    /// The size of the queue.
    pub fn len(&self) -> usize {
        self.global.len()
    }

    /// If the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn stream_id_from_different_threads() {
        let current = StreamId::current();

        let thread1 = std::thread::spawn(|| (StreamId::current(), StreamId::current()));
        let thread2 = std::thread::spawn(StreamId::current);

        let (stream_1, stream_11) = thread1.join().unwrap();
        let stream_2 = thread2.join().unwrap();

        assert_ne!(current, stream_1, "Should be different from thread 1");
        assert_ne!(current, stream_2, "Should be different from thread 2");
        assert_ne!(
            stream_1, stream_2,
            "Should be different from different threads"
        );
        assert_eq!(
            stream_1, stream_11,
            "Should be the same, since same thread."
        );
    }
}
