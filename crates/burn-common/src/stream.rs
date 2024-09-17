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
