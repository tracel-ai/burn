use super::{DataLoader, DataLoaderIterator, DynDataLoader, Progress};
use std::sync::mpsc;
use std::thread;

const MAX_QUEUED_ITEMS: usize = 100;

/// A multi-threaded data loader that can be used to iterate over a dataset.
pub struct MultiThreadDataLoader<O> {
    dataloaders: Vec<Box<dyn DynDataLoader<O>>>,
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
    progresses: Vec<Progress>,
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
    pub fn new(dataloaders: Vec<Box<dyn DynDataLoader<O>>>) -> Self {
        Self { dataloaders }
    }
}

impl<O> DataLoader<O> for MultiThreadDataLoader<O>
where
    O: Send + 'static + std::fmt::Debug,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        let (sender, receiver) = mpsc::sync_channel::<Message<O>>(MAX_QUEUED_ITEMS);

        let mut progresses = Vec::with_capacity(self.dataloaders.len());

        let handlers: Vec<_> = self
            .dataloaders
            .iter()
            .enumerate()
            .map(|(index, dataloader)| {
                let dataloader_cloned = dataloader.clone_dyn();
                let sender_cloned = sender.clone();
                progresses.push(Progress::new(0, dataloader_cloned.num_items()));

                thread::spawn(move || {
                    let mut iterator = dataloader_cloned.iter();
                    while let Some(item) = iterator.next() {
                        let progress = iterator.progress();

                        match sender_cloned.send(Message::Batch(index, item, progress)) {
                            Ok(_) => {}
                            // The receiver is probably gone, no need to panic, just need to stop
                            // iterating.
                            Err(_) => return,
                        };
                    }
                    // Same thing.
                    sender_cloned.send(Message::Done).ok();
                })
            })
            .collect();

        Box::new(MultiThreadsDataloaderIterator::new(
            receiver, handlers, progresses,
        ))
    }

    fn num_items(&self) -> usize {
        self.dataloaders.iter().map(|dl| dl.num_items()).sum()
    }
}

impl<O> MultiThreadsDataloaderIterator<O> {
    pub fn new(
        receiver: mpsc::Receiver<Message<O>>,
        workers: Vec<thread::JoinHandle<()>>,
        progresses: Vec<Progress>,
    ) -> Self {
        MultiThreadsDataloaderIterator {
            num_done: 0,
            workers,
            receiver,
            progresses,
        }
    }
}
impl<O: std::fmt::Debug> DataLoaderIterator<O> for MultiThreadsDataloaderIterator<O> {
    fn progress(&self) -> Progress {
        let mut items_total = 0;
        let mut items_processed = 0;

        for progress in self.progresses.iter() {
            items_total += progress.items_total;
            items_processed += progress.items_processed;
        }

        Progress::new(items_processed, items_total)
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
                    if let Some(current) = self.progresses.get_mut(index) {
                        *current = progress;
                    }
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
