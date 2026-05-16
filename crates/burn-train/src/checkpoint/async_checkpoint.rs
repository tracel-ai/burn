use super::{Checkpointer, CheckpointerError};
use crate::Interrupter;
use burn_core::{record::Record, tensor::Device};
use std::sync::mpsc;

enum Message<R> {
    Restore(
        usize,
        Device,
        mpsc::SyncSender<Result<R, CheckpointerError>>,
        Option<Interrupter>,
    ),
    Save(usize, R, Option<Interrupter>),
    Delete(usize, Option<Interrupter>),
    End,
}

#[derive(new)]
struct CheckpointerThread<C, R> {
    checkpointer: C,
    receiver: mpsc::Receiver<Message<R>>,
}

impl<C, R> CheckpointerThread<C, R>
where
    C: Checkpointer<R>,
    R: Record,
{
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::Restore(epoch, device, callback, interrupter) => {
                    let record = self.checkpointer.restore(epoch, &device);
                    callback.send(record).unwrap_or_else(|err| {
                        interrupter.map_or_else(
                            || {
                                panic!(
                                    "Error when sending response through callback channel: {err}"
                                )
                            },
                            |int| int.stop(Some(&err.to_string())),
                        )
                    });
                }
                Message::Save(epoch, state, interrupter) => {
                    self.checkpointer.save(epoch, state).unwrap_or_else(|err| {
                        interrupter.map_or_else(
                            || panic!("Error when saving the state: {err}"),
                            |int| int.stop(Some(&err.to_string())),
                        )
                    });
                }
                Message::Delete(epoch, interrupter) => {
                    self.checkpointer.delete(epoch).unwrap_or_else(|err| {
                        interrupter.map_or_else(
                            || panic!("Error when deleting the state: {err}"),
                            |int| int.stop(Some(&err.to_string())),
                        )
                    });
                }

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
    interrupter: Option<Interrupter>,
}

impl<R> AsyncCheckpointer<R>
where
    R: Record + 'static,
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

        Self {
            sender,
            handler,
            interrupter: None,
        }
    }

    /// Assign a handle used to interrupt training in case of checkpointing error.
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.interrupter = Some(interrupter);
        self
    }
}

impl<R> Checkpointer<R> for AsyncCheckpointer<R>
where
    R: Record + 'static,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        self.sender
            .send(Message::Save(epoch, record, self.interrupter.clone()))
            .expect("Can send message to checkpointer thread.");

        Ok(())
    }

    fn restore(&self, epoch: usize, device: &Device) -> Result<R, CheckpointerError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Message::Restore(
                epoch,
                device.clone(),
                sender,
                self.interrupter.clone(),
            ))
            .map_err(|e| CheckpointerError::Unknown(e.to_string()))?;

        if let Ok(record) = receiver.recv() {
            return record;
        };

        Err(CheckpointerError::Unknown("Channel error.".to_string()))
    }

    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
        self.sender
            .send(Message::Delete(epoch, self.interrupter.clone()))
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
