use super::Logger;
use std::sync::{mpsc, Mutex};

enum Message<T> {
    Log(T),
}

pub struct AsyncLogger<T> {
    sender: mpsc::Sender<Message<T>>,
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
            }
        }
    }
}

impl<T: Send + Sync + 'static> AsyncLogger<T> {
    pub fn new(logger: Box<dyn Logger<T>>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let thread = LoggerThread::new(Mutex::new(logger), receiver);

        std::thread::spawn(move || thread.run());

        Self { sender }
    }
}

impl<T: Send> Logger<T> for AsyncLogger<T> {
    fn log(&mut self, item: T) {
        self.sender.send(Message::Log(item)).unwrap();
    }
}
