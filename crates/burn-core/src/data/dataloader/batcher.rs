/// A trait for batching items of type `I` into items of type `O`.
pub trait Batcher<I, O>: Send + Sync {
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
