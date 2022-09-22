pub trait Logger<T>: Send {
    fn log(&mut self, item: T);
    fn clear(&mut self);
}

pub trait TrainValidLogger<T, V>: Send {
    fn log_train(&mut self, item: T);
    fn log_valid(&mut self, item: V);
    fn clear_train(&mut self);
    fn clear_valid(&mut self);
}
