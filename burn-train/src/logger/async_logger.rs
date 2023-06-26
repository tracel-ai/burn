use super::Logger;
use std::sync::{mpsc, Mutex};

enum Message<T> {
    Log(T),
    End,
}
/// Async logger.
pub struct AsyncLogger<T> {
    sender: mpsc::Sender<Message<T>>,
    handler: Option<std::thread::JoinHandle<()>>,
}

#[derive(new)]
struct LoggerThread<T> {
    logger: Mutex<Box<dyn Logger<T>>>,
    receiver: mpsc::Receiver<Message<T>>,
}

impl<T> LoggerThread<T> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::Log(item) => {
                    let mut logger = self.logger.lock().unwrap();
                    logger.log(item);
                }
                Message::End => {
                    return;
                }
            }
        }
    }
}

impl<T: Send + Sync + 'static> AsyncLogger<T> {
    /// Create a new async logger.
    pub fn new(logger: Box<dyn Logger<T>>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let thread = LoggerThread::new(Mutex::new(logger), receiver);

        let handler = Some(std::thread::spawn(move || thread.run()));

        Self { sender, handler }
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
