use super::DataLoader;
use std::sync::{mpsc, Arc};
use std::thread;

static MAX_QUEUED_ITEMS: usize = 100;

pub struct MultiThreadDataLoader<O> {
    dataloaders: Vec<Arc<dyn DataLoader<O> + Send + Sync>>,
}

#[derive(Debug)]
pub enum Message<O> {
    Batch(O),
    Done,
}

struct MultiThreadsDataloaderIterator<O> {
    num_done: usize,
    workers: Vec<thread::JoinHandle<()>>,
    receiver: mpsc::Receiver<Message<O>>,
}

impl<O> MultiThreadDataLoader<O> {
    pub fn new(dataloaders: Vec<Arc<dyn DataLoader<O> + Send + Sync>>) -> Self {
        Self { dataloaders }
    }
}

impl<O> DataLoader<O> for MultiThreadDataLoader<O>
where
    O: Send + 'static + std::fmt::Debug,
{
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = O> + 'a> {
        let (sender, receiver) = mpsc::sync_channel::<Message<O>>(MAX_QUEUED_ITEMS);

        let handlers: Vec<_> = self
            .dataloaders
            .clone()
            .into_iter()
            .map(|dataloader| {
                let dataloader_cloned = dataloader.clone();
                let sender_cloned = sender.clone();

                thread::spawn(move || {
                    for item in dataloader_cloned.iter() {
                        sender_cloned.send(Message::Batch(item)).unwrap();
                    }
                    sender_cloned.send(Message::Done).unwrap();
                })
            })
            .collect();

        Box::new(MultiThreadsDataloaderIterator::new(receiver, handlers))
    }

    fn len(&self) -> usize {
        let mut len = 0;
        for dataloader in self.dataloaders.iter() {
            len += dataloader.len();
        }
        len
    }
}

impl<O> MultiThreadsDataloaderIterator<O> {
    pub fn new(receiver: mpsc::Receiver<Message<O>>, workers: Vec<thread::JoinHandle<()>>) -> Self {
        MultiThreadsDataloaderIterator {
            num_done: 0,
            workers,
            receiver,
        }
    }
}

impl<O: std::fmt::Debug> Iterator for MultiThreadsDataloaderIterator<O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        if self.workers.len() == 0 {
            return None;
        }

        loop {
            let item = self.receiver.recv();
            let item = item.unwrap();

            match item {
                Message::Batch(item) => return Some(item),
                Message::Done => {
                    self.num_done += 1;
                }
            };

            if self.num_done == self.workers.len() {
                while let Some(worker) = self.workers.pop() {
                    worker.join().unwrap();
                }
                return None;
            }
        }
    }
}
