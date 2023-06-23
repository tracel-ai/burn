use super::{LearnerCallback, LearnerItem};
use std::{
    sync::{mpsc, Mutex},
    thread::JoinHandle,
};

enum Message<T, V> {
    LogTrain(LearnerItem<T>),
    LogValid(LearnerItem<V>),
    ClearTrain(usize),
    ClearValid(usize),
    End,
}

/// Async trainer callback tracker.
pub struct AsyncTrainerCallback<T, V> {
    sender: mpsc::Sender<Message<T, V>>,
    handler: Option<JoinHandle<()>>,
}

#[derive(new)]
struct CallbackThread<T, V> {
    callback: Mutex<Box<dyn LearnerCallback<T, V>>>,
    receiver: mpsc::Receiver<Message<T, V>>,
}

impl<T, V> CallbackThread<T, V> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::LogTrain(item) => {
                    let mut callback = self.callback.lock().unwrap();
                    callback.on_train_item(item);
                }
                Message::ClearTrain(epoch) => {
                    let mut callback = self.callback.lock().unwrap();
                    callback.on_train_end_epoch(epoch);
                }
                Message::LogValid(item) => {
                    let mut callback = self.callback.lock().unwrap();
                    callback.on_valid_item(item);
                }
                Message::ClearValid(epoch) => {
                    let mut callback = self.callback.lock().unwrap();
                    callback.on_valid_end_epoch(epoch);
                }
                Message::End => {
                    return;
                }
            }
        }
    }
}

impl<T: Send + Sync + 'static, V: Send + Sync + 'static> AsyncTrainerCallback<T, V> {
    /// Create a new async trainer callback.
    pub fn new(callback: Box<dyn LearnerCallback<T, V>>) -> Self {
        let (sender, receiver) = mpsc::channel();
        let thread = CallbackThread::new(Mutex::new(callback), receiver);

        let handler = std::thread::spawn(move || thread.run());
        let handler = Some(handler);

        Self { sender, handler }
    }
}

impl<T: Send, V: Send> LearnerCallback<T, V> for AsyncTrainerCallback<T, V> {
    fn on_train_item(&mut self, item: LearnerItem<T>) {
        self.sender.send(Message::LogTrain(item)).unwrap();
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        self.sender.send(Message::LogValid(item)).unwrap();
    }

    fn on_train_end_epoch(&mut self, epoch: usize) {
        self.sender.send(Message::ClearTrain(epoch)).unwrap();
    }

    fn on_valid_end_epoch(&mut self, epoch: usize) {
        self.sender.send(Message::ClearValid(epoch)).unwrap();
    }
}

impl<T, V> Drop for AsyncTrainerCallback<T, V> {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
