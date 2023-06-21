use super::{DataLoader, DataLoaderIterator, Progress};
use std::collections::HashMap;
use std::sync::{mpsc, Arc};
use std::thread;

const MAX_QUEUED_ITEMS: usize = 100;

/// A multi-threaded data loader that can be used to iterate over a dataset.
pub struct MultiThreadDataLoader<O> {
    dataloaders: Vec<Arc<dyn DataLoader<O> + Send + Sync>>,
}

/// A message that can be sent between threads.
#[derive(Debug)]
pub enum Message<O> {
    /// A batch of items.
    Batch(usize, O, Progress),

    /// The thread is done.
    Done,
}

struct MultiThreadsDataloaderIterator<O> {
    num_done: usize,
    workers: Vec<thread::JoinHandle<()>>,
    receiver: mpsc::Receiver<Message<O>>,
    progresses: HashMap<usize, Progress>,
}

impl<O> MultiThreadDataLoader<O> {
    /// Creates a new multi-threaded data loader.
    ///
    /// # Arguments
    ///
    /// * `dataloaders` - The data loaders.
    ///
    /// # Returns
    ///
    /// The multi-threaded data loader.
    pub fn new(dataloaders: Vec<Arc<dyn DataLoader<O> + Send + Sync>>) -> Self {
        Self { dataloaders }
    }
}

impl<O> DataLoader<O> for MultiThreadDataLoader<O>
where
    O: Send + 'static + std::fmt::Debug,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        let (sender, receiver) = mpsc::sync_channel::<Message<O>>(MAX_QUEUED_ITEMS);

        let handlers: Vec<_> = self
            .dataloaders
            .clone()
            .into_iter()
            .enumerate()
            .map(|(index, dataloader)| {
                let dataloader_cloned = dataloader;
                let sender_cloned = sender.clone();

                thread::spawn(move || {
                    let mut iterator = dataloader_cloned.iter();
                    while let Some(item) = iterator.next() {
                        let progress = iterator.progress();
                        sender_cloned
                            .send(Message::Batch(index, item, progress))
                            .unwrap();
                    }
                    sender_cloned.send(Message::Done).unwrap();
                })
            })
            .collect();

        Box::new(MultiThreadsDataloaderIterator::new(receiver, handlers))
    }
}

impl<O> MultiThreadsDataloaderIterator<O> {
    pub fn new(receiver: mpsc::Receiver<Message<O>>, workers: Vec<thread::JoinHandle<()>>) -> Self {
        MultiThreadsDataloaderIterator {
            num_done: 0,
            workers,
            receiver,
            progresses: HashMap::new(),
        }
    }
}
impl<O: std::fmt::Debug> DataLoaderIterator<O> for MultiThreadsDataloaderIterator<O> {
    fn progress(&self) -> Progress {
        let mut items_total = 0;
        let mut items_processed = 0;

        for progress in self.progresses.values() {
            items_total += progress.items_total;
            items_processed += progress.items_processed;
        }

        Progress {
            items_processed,
            items_total,
        }
    }
}

impl<O: std::fmt::Debug> Iterator for MultiThreadsDataloaderIterator<O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        if self.workers.is_empty() {
            return None;
        }

        loop {
            let item = self.receiver.recv();
            let item = item.unwrap();

            match item {
                Message::Batch(index, item, progress) => {
                    self.progresses.insert(index, progress);
                    return Some(item);
                }
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
