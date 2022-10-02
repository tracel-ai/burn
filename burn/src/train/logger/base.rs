pub trait Logger<T>: Send {
    fn log(&mut self, item: T);
}

pub trait LoggerBackend {
    type Logger<T>: Logger<T>;

    fn create<T>(&self, epoch: usize) -> Self::Logger<T>;
}
