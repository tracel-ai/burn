use super::TrainValidLogger;
use std::sync::{mpsc, Mutex};

enum Message<T, V> {
    LogTrain(T),
    LogValid(V),
    ClearTrain,
    ClearValid,
}

pub struct AsyncTrainValidLogger<T, V> {
    sender: mpsc::Sender<Message<T, V>>,
}

#[derive(new)]
struct LoggerThread<T, V> {
    logger: Mutex<Box<dyn TrainValidLogger<T, V>>>,
    receiver: mpsc::Receiver<Message<T, V>>,
}

impl<T, V> LoggerThread<T, V> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::LogTrain(item) => {
                    let mut logger = self.logger.lock().unwrap();
                    logger.log_train(item);
                }
                Message::ClearTrain => {
                    let mut logger = self.logger.lock().unwrap();
                    logger.clear_train();
                }
                Message::LogValid(item) => {
                    let mut logger = self.logger.lock().unwrap();
                    logger.log_valid(item);
                }
                Message::ClearValid => {
                    let mut logger = self.logger.lock().unwrap();
                    logger.clear_valid();
                }
            }
        }
    }
}

impl<T: Send + Sync + 'static, V: Send + Sync + 'static> AsyncTrainValidLogger<T, V> {
    pub fn new(logger: Box<dyn TrainValidLogger<T, V>>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let thread = LoggerThread::new(Mutex::new(logger), receiver);

        std::thread::spawn(move || thread.run());

        Self { sender }
    }
}

impl<T: Send, V: Send> TrainValidLogger<T, V> for AsyncTrainValidLogger<T, V> {
    fn log_train(&mut self, item: T) {
        self.sender.send(Message::LogTrain(item)).unwrap();
    }

    fn log_valid(&mut self, item: V) {
        self.sender.send(Message::LogValid(item)).unwrap();
    }

    fn clear_train(&mut self) {
        self.sender.send(Message::ClearTrain).unwrap();
    }

    fn clear_valid(&mut self) {
        self.sender.send(Message::ClearValid).unwrap();
    }
}
