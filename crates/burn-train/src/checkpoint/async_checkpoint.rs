use super::{Checkpointer, CheckpointerError};
use crate::Interrupter;
use burn_core::{record::Record, tensor::backend::Backend};
use std::sync::mpsc;

enum Message<R, B: Backend> {
    Restore(
        usize,
        B::Device,
        mpsc::SyncSender<Result<R, CheckpointerError>>,
        Option<Interrupter>,
    ),
    Save(usize, R, Option<Interrupter>),
    Delete(usize, Option<Interrupter>),
    End,
}

#[derive(new)]
struct CheckpointerThread<C, R, B: Backend> {
    checkpointer: C,
    receiver: mpsc::Receiver<Message<R, B>>,
}

impl<C, R, B> CheckpointerThread<C, R, B>
where
    C: Checkpointer<R, B>,
    R: Record<B>,
    B: Backend,
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
pub struct AsyncCheckpointer<Record, B: Backend> {
    sender: mpsc::SyncSender<Message<Record, B>>,
    handler: Option<std::thread::JoinHandle<()>>,
    interrupter: Option<Interrupter>,
}

impl<R, B> AsyncCheckpointer<R, B>
where
    R: Record<B> + 'static,
    B: Backend,
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
        C: Checkpointer<R, B> + Send + 'static,
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

impl<R, B> Checkpointer<R, B> for AsyncCheckpointer<R, B>
where
    R: Record<B> + 'static,
    B: Backend,
{
    fn save(&self, epoch: usize, record: R) -> Result<(), CheckpointerError> {
        self.sender
            .send(Message::Save(epoch, record, self.interrupter.clone()))
            .expect("Can send message to checkpointer thread.");

        Ok(())
    }

    fn restore(&self, epoch: usize, device: &B::Device) -> Result<R, CheckpointerError> {
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

impl<E, B> Drop for AsyncCheckpointer<E, B>
where
    B: Backend,
{
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
