use super::{Checkpointer, CheckpointerError};
use burn_core::record::Record;
use std::sync::{mpsc, Arc};

enum Message<R> {
    Save(usize, R),
    End,
}

#[derive(new)]
struct CheckpointerThread<R> {
    checkpointer: Arc<dyn Checkpointer<R> + Send + Sync>,
    receiver: mpsc::Receiver<Message<R>>,
}

impl<R: Record> CheckpointerThread<R> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::Save(epoch, state) => self.checkpointer.save(epoch, state).unwrap(),
                Message::End => {
                    return;
                }
            };
        }
    }
}

/// Async checkpointer.
pub struct AsyncCheckpointer<E> {
    checkpointer: Arc<dyn Checkpointer<E> + Send + Sync>,
    sender: mpsc::SyncSender<Message<E>>,
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
    pub fn new(checkpointer: Arc<dyn Checkpointer<R> + Send + Sync>) -> Self {
        // Only on checkpoint can be done in advance.
        let (sender, receiver) = mpsc::sync_channel(0);
        let thread = CheckpointerThread::new(checkpointer.clone(), receiver);
        let handler = Some(std::thread::spawn(move || thread.run()));

        Self {
            checkpointer,
            sender,
            handler,
        }
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
        self.checkpointer.restore(epoch)
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
