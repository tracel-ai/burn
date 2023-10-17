use super::{Checkpointer, CheckpointerError};
use burn_core::record::Record;
use std::sync::mpsc;

enum Message<R> {
    Restore(usize, mpsc::SyncSender<Result<R, CheckpointerError>>),
    Save(usize, R),
    Delete(usize),
    End,
}

#[derive(new)]
struct CheckpointerThread<C, R> {
    checkpointer: C,
    receiver: mpsc::Receiver<Message<R>>,
}

impl<C: Checkpointer<R>, R: Record> CheckpointerThread<C, R> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::Restore(epoch, sender) => {
                    let record = self.checkpointer.restore(epoch);
                    sender.send(record).unwrap();
                }
                Message::Save(epoch, state) => self.checkpointer.save(epoch, state).unwrap(),
                Message::Delete(epoch) => self.checkpointer.delete(epoch).unwrap(),
                Message::End => {
                    return;
                }
            };
        }
    }
}

/// Async checkpointer.
pub struct AsyncCheckpointer<Record> {
    sender: mpsc::SyncSender<Message<Record>>,
    handler: Option<std::thread::JoinHandle<()>>,
}

impl<R: Record + 'static> AsyncCheckpointer<R> {
    /// Create a new async checkpointer.
    ///
    /// # Arguments
    ///
    /// * `checkpointer` - The checkpointer.
    ///
    /// # Returns
    ///
    /// The async checkpointer.
    pub fn new<C>(checkpointer: C) -> Self
    where
        C: Checkpointer<R> + Send + 'static,
    {
        // Only on checkpoint can be done in advance.
        let (sender, receiver) = mpsc::sync_channel(0);
        let thread = CheckpointerThread::new(checkpointer, receiver);
        let handler = Some(std::thread::spawn(move || thread.run()));

        Self { sender, handler }
    }
}

impl<R> Checkpointer<R> for AsyncCheckpointer<R>
where
    R: Record + 'static,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        self.sender.send(Message::Save(epoch, record)).unwrap();

        Ok(())
    }

    fn restore(&self, epoch: usize) -> Result<R, CheckpointerError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Message::Restore(epoch, sender))
            .map_err(|e| CheckpointerError::Unknown(e.to_string()))?;

        if let Ok(record) = receiver.recv() {
            return record;
        };

        Err(CheckpointerError::Unknown("Channel error.".to_string()))
    }

    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
        self.sender
            .send(Message::Delete(epoch))
            .map_err(|e| CheckpointerError::Unknown(e.to_string()))?;

        Ok(())
    }
}

impl<E> Drop for AsyncCheckpointer<E> {
    fn drop(&mut self) {
        self.sender.send(Message::End).unwrap();
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler.join().unwrap();
        }
    }
}
