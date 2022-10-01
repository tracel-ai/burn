use super::{Checkpointer, CheckpointerError};
use crate::module::State;
use burn_tensor::Element;
use std::sync::{mpsc, Arc};

enum Message<E> {
    Save(usize, State<E>),
}

#[derive(new)]
struct CheckpointerThread<T> {
    checkpointer: Arc<dyn Checkpointer<T> + Send + Sync>,
    receiver: mpsc::Receiver<Message<T>>,
}

impl<T> CheckpointerThread<T> {
    fn run(self) {
        for item in self.receiver.iter() {
            match item {
                Message::Save(epoch, state) => self.checkpointer.save(epoch, state).unwrap(),
            };
        }
    }
}

pub struct AsyncCheckpointer<E> {
    checkpointer: Arc<dyn Checkpointer<E> + Send + Sync>,
    sender: mpsc::SyncSender<Message<E>>,
}

impl<E: Element + 'static> AsyncCheckpointer<E> {
    pub fn new(checkpointer: Arc<dyn Checkpointer<E> + Send + Sync>) -> Self {
        // Only on checkpoint can be done in advance.
        let (sender, receiver) = mpsc::sync_channel(0);
        let thread = CheckpointerThread::new(checkpointer.clone(), receiver);

        std::thread::spawn(move || thread.run());

        Self {
            checkpointer,
            sender,
        }
    }
}

impl<E> Checkpointer<E> for AsyncCheckpointer<E>
where
    E: Element + Sync + 'static,
{
    fn save(&self, epoch: usize, state: State<E>) -> Result<(), CheckpointerError> {
        self.sender.send(Message::Save(epoch, state)).unwrap();

        Ok(())
    }

    fn restore(&self, epoch: usize) -> Result<State<E>, CheckpointerError> {
        self.checkpointer.restore(epoch)
    }
}
