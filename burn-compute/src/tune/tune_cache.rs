use core::marker::PhantomData;
use hashbrown::HashMap;

use crate::server::ComputeServer;

use super::AutotuneOperation;
use super::Operation;

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub struct TuneCache<S> {
    cache: HashMap<String, usize>,
    _server: PhantomData<S>,
}

impl<S: ComputeServer> TuneCache<S> {
    pub(crate) fn new() -> Self {
        TuneCache {
            cache: HashMap::new(),
            _server: PhantomData,
        }
    }

    pub(crate) fn try_cache(
        &self,
        autotune_operation: &Box<dyn AutotuneOperation<S>>,
    ) -> Option<Operation<S>> {
        let index = self.cache.get(&autotune_operation.key());
        if let Some(&i) = index {
            return Some(autotune_operation.fastest(i));
        }
        None
    }

    pub(crate) fn cache_insert(&mut self, key: String, fastest_index: usize) {
        self.cache.insert(key, fastest_index);
    }
}
