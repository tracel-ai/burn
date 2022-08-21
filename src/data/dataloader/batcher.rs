pub trait Batcher<I, O>: Send {
    fn batch(&self, items: Vec<I>) -> O;
}
