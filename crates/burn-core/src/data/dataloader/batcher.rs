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
}

/// A super trait for [batcher](Batcher) that allows it to be cloned dynamically.
///
/// Any batcher that implements [Clone] should also implement this automatically.
pub trait DynBatcher<I, O>: Send + Batcher<I, O> {
    /// Clone the batcher and returns a new one.
    fn clone_dyn(&self) -> Box<dyn DynBatcher<I, O>>;
}

impl<B, I, O> DynBatcher<I, O> for B
where
    B: Batcher<I, O> + Clone + 'static,
{
    fn clone_dyn(&self) -> Box<dyn DynBatcher<I, O>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
#[derive(new, Clone)]
pub struct TestBatcher;

#[cfg(test)]
impl<I> Batcher<I, Vec<I>> for TestBatcher {
    fn batch(&self, items: Vec<I>) -> Vec<I> {
        items
    }
}
