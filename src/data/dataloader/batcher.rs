pub trait Batcher<I, O> {
    fn batch(&self, items: Vec<I>) -> O;
}
