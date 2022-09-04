pub trait Batcher<I, O>: Send + Sync {
    fn batch(&self, items: Vec<I>) -> O;
}

#[cfg(test)]
#[derive(new)]
pub struct TestBatcher;
#[cfg(test)]
impl<I> Batcher<I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>) -> Vec<I> {
        items
    }
}
