use std::{
    pin::Pin,
    task::{Context, Poll, Waker},
};

use burn_std::id::StreamId;

/// A future polled on a server stream: once when it is made, then whenever it is polled again.
///
/// A backend's read launches its copy to the host when first polled, not when made. Polled first
/// on a runtime thread, it would run on that thread's stream, which no graph replay is ordered
/// against.
pub struct StreamFuture<F: Future> {
    stream: StreamId,
    future: F,
    ready: Option<F::Output>,
}

impl<F: Future + Unpin> StreamFuture<F>
where
    F::Output: Unpin,
{
    /// `future` on `stream`, polled once now so what it launches joins the stream in order.
    pub fn start(stream: StreamId, future: F) -> Self {
        let mut started = Self {
            stream,
            future,
            ready: None,
        };
        if let Poll::Ready(output) = started.poll_on_stream(&mut Context::from_waker(Waker::noop()))
        {
            started.ready = Some(output);
        }
        started
    }

    fn poll_on_stream(&mut self, cx: &mut Context<'_>) -> Poll<F::Output> {
        let future = &mut self.future;
        self.stream.executes(|| Pin::new(future).poll(cx))
    }
}

impl<F: Future + Unpin> Future for StreamFuture<F>
where
    F::Output: Unpin,
{
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.ready.take() {
            Some(output) => Poll::Ready(output),
            None => this.poll_on_stream(cx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn it_is_polled_on_its_stream_when_made_and_after() {
        let stream = StreamId {
            value: u64::MAX - 1,
        };
        let polls = RefCell::new(Vec::new());
        let mut future = StreamFuture::start(
            stream,
            std::future::poll_fn(|cx| {
                polls.borrow_mut().push(StreamId::current());
                if polls.borrow().len() < 2 {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(())
            }),
        );
        assert_eq!(*polls.borrow(), [stream], "not polled when made");

        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut future).poll(&mut cx).is_ready());
        assert_eq!(*polls.borrow(), [stream, stream]);
    }

    #[test]
    fn an_output_ready_when_made_is_kept_for_the_first_poll() {
        let mut future = StreamFuture::start(StreamId { value: 0 }, std::future::ready(7));
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(Pin::new(&mut future).poll(&mut cx), Poll::Ready(7));
    }
}
