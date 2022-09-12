pub trait Logger<T>: Send {
    fn log(&mut self, item: T);
    fn clear(&mut self);
}
