use super::Logger;
use std::sync::mpsc;

enum Message<T> {
    Log(T),
    End,
    Sync(mpsc::Sender<()>),
}
/// Async logger.
pub struct AsyncLogger<T> {
    sender: mpsc::Sender<Message<T>>,
    handler: Option<std::thread::JoinHandle<()>>,
}

#[derive(new)]
struct LoggerThread<T, L: Logger<T>> {
    logger: L,
    receiver: mpsc::Receiver<Message<T>>,
}

impl<T, L> LoggerThread<T, L>
where
    L: Logger<T>,
{
    fn run(mut self) {
        for item in self.receiver.iter() {
            match item {
                Message::Log(item) => {
                    self.logger.log(item);
                }
                Message::End => {
                    return;
                }
                Message::Sync(callback) => {
                    callback.send(()).unwrap();
                }
            }
        }
    }
}

impl<T: Send + Sync + 'static> AsyncLogger<T> {
    /// Create a new async logger.
    pub fn new<L>(logger: L) -> Self
    where
        L: Logger<T> + 'static,
    {
        let (sender, receiver) = mpsc::channel();
        let thread = LoggerThread::new(logger, receiver);

        let handler = Some(std::thread::spawn(move || thread.run()));

        Self { sender, handler }
    }

    /// Sync the async logger.
    pub(crate) fn sync(&self) {
        let (sender, receiver) = mpsc::channel();

        self.sender.send(Message::Sync(sender)).unwrap();

        receiver
            .recv()
            .expect("Should sync, otherwise the thread is dead.");
    }
}

impl<T: Send> Logger<T> for AsyncLogger<T> {
    fn log(&mut self, item: T) {
        self.sender.send(Message::Log(item)).unwrap();
    }
}

impl<T> Drop for AsyncLogger<T> {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
