use core::marker::PhantomData;

use hashbrown::HashMap;

use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub struct TuneCache<S> {
    cache: HashMap<AutotuneKey, usize>,
    _server: PhantomData<S>,
}

impl<S> TuneCache<S> {
    pub(crate) fn new() -> Self {
        TuneCache {
            cache: HashMap::new(),
            _server: PhantomData,
        }
    }

    #[allow(clippy::borrowed_box)]
    pub(crate) fn try_cache(
        &self,
        autotune_operation: &Box<dyn AutotuneOperationSet<S>>,
    ) -> Option<Box<dyn AutotuneOperation<S>>> {
        let index = self.cache.get(&autotune_operation.key());
        if let Some(&i) = index {
            return Some(autotune_operation.fastest(i));
        }
        None
    }

    pub(crate) fn cache_insert(&mut self, key: AutotuneKey, fastest_index: usize) {
        self.cache.insert(key, fastest_index);
    }
}
