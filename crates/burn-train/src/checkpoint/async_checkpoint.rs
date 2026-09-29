use super::{Checkpoint, Checkpointer, CheckpointerError};
use crate::Interrupter;
use std::sync::mpsc;

enum Message<R> {
    Restore(usize, mpsc::SyncSender<Result<R, CheckpointerError>>),
    Save(usize, R),
    Delete(usize),
    Interrupter(Interrupter),
    End,
}

struct CheckpointerThread<C, R> {
    checkpointer: C,
    receiver: mpsc::Receiver<Message<R>>,
    interrupter: Option<Interrupter>,
}

impl<C, R> CheckpointerThread<C, R>
where
    C: Checkpointer<R>,
    R: Checkpoint,
{
    fn new(checkpointer: C, receiver: mpsc::Receiver<Message<R>>) -> Self {
        Self {
            checkpointer,
            receiver,
            interrupter: None,
        }
    }

    fn run(mut self) {
        while let Ok(item) = self.receiver.recv() {
            match item {
                Message::Restore(epoch, callback) => {
                    let record = self.checkpointer.restore(epoch);
                    if let Err(err) = callback.send(record) {
                        self.fail("Error when sending response through callback channel", err);
                    }
                }
                Message::Save(epoch, state) => {
                    if let Err(err) = self.checkpointer.save(epoch, state) {
                        self.fail("Error when saving the state", err);
                    }
                }
                Message::Delete(epoch) => {
                    if let Err(err) = self.checkpointer.delete(epoch) {
                        self.fail("Error when deleting the state", err);
                    }
                }
                Message::Interrupter(interrupter) => {
                    self.interrupter = Some(interrupter);
                }
                Message::End => {
                    return;
                }
            };
        }
    }

    /// Interrupt training with `err`, or panic when there is no interrupter to report it to.
    fn fail(&self, context: &str, err: impl core::fmt::Display) {
        match &self.interrupter {
            Some(interrupter) => interrupter.stop(Some(&err.to_string())),
            None => panic!("{context}: {err}"),
        }
    }
}

/// Async checkpointer.
pub struct AsyncCheckpointer<R> {
    sender: mpsc::SyncSender<Message<R>>,
    handler: Option<std::thread::JoinHandle<()>>,
}

impl<R> AsyncCheckpointer<R>
where
    R: Checkpoint,
{
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

    /// Assign a handle used to interrupt training in case of checkpointing error.
    pub fn with_interrupter(self, interrupter: Interrupter) -> Self {
        self.sender
            .send(Message::Interrupter(interrupter))
            .expect("Can send message to checkpointer thread.");
        self
    }
}

impl<R> Checkpointer<R> for AsyncCheckpointer<R>
where
    R: Checkpoint,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        self.sender
            .send(Message::Save(epoch, record))
            .expect("Can send message to checkpointer thread.");

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
        self.sender
            .send(Message::End)
            .expect("Can send the end message to the checkpointer thread.");
        let handler = self.handler.take();

        if let Some(handler) = handler {
            handler
                .join()
                .expect("The checkpointer thread should stop.");
        }
    }
}
