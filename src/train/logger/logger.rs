pub struct LogItem<T> {
    pub epoch: Option<usize>,
    pub epoch_total: Option<usize>,
    pub iteration: usize,
    pub iteration_total: usize,
    pub item: T,
}

impl<T> LogItem<T> {
    pub fn new(item: T) -> Self {
        Self {
            epoch: None,
            epoch_total: None,
            iteration: 0,
            iteration_total: 0,
            item,
        }
    }

    pub fn iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    pub fn iteration_total(mut self, iteration: usize) -> Self {
        self.iteration_total = iteration;
        self
    }

    pub fn epoch(mut self, epoch: usize) -> Self {
        self.epoch = Some(epoch);
        self
    }

    pub fn epoch_total(mut self, epoch: usize) -> Self {
        self.epoch_total = Some(epoch);
        self
    }
}

pub trait Logger<T>: Send {
    fn log(&mut self, item: LogItem<T>);
    fn clear(&mut self);
}
