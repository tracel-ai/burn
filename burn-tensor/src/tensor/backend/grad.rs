pub trait Gradients: Send + Sync {
    fn empty() -> Self;
    fn get<V: 'static>(&self, id: &str) -> Option<&V>;
    fn register<V>(&mut self, id: String, value: V)
    where
        V: std::fmt::Debug + 'static + Send + Sync;
}
