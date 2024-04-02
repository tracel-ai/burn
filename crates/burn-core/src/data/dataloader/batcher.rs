/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<I, O>: Send {
    /// Batches the given items.
    ///
    /// # Arguments
    ///
    /// * `items` - The items to batch.
    ///
    /// # Returns
    ///
    /// The batched items.
    fn batch(&self, items: Vec<I>) -> O;
    /// Create a new batcher similar to this one.
    fn new_like(&self) -> Box<dyn Batcher<I, O>>;
}

#[cfg(test)]
#[derive(new)]
pub struct TestBatcher;
#[cfg(test)]
impl<I> Batcher<I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>) -> Vec<I> {
        items
    }

    fn new_like(&self) -> Box<dyn Batcher<I, Vec<I>>> {
        Box::new(TestBatcher)
    }
}
