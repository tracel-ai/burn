pub trait Batcher<I, O>: Send + Sync {
    fn batch(&self, items: Vec<I>) -> O;
}
