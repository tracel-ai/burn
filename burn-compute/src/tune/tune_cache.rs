use core::marker::PhantomData;

use hashbrown::HashMap;

use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;

/// Use to find and reuse the best kernel for some input
#[derive(Debug, Default)]
pub(crate) struct TuneCache<S> {
    cache: HashMap<AutotuneKey, usize>,
    _server: PhantomData<S>,
}

/// Result of the cache try
pub enum TuneCacheResult {
    /// An operation is found and given
    Hit(Box<dyn AutotuneOperation>),
    /// No operation is found and the set is given back for ownership
    Miss(Box<dyn AutotuneOperationSet>),
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
        autotune_operation_set: Box<dyn AutotuneOperationSet>,
    ) -> TuneCacheResult {
        let index = self.cache.get(&autotune_operation_set.key());
        if let Some(&i) = index {
            return TuneCacheResult::Hit(autotune_operation_set.fastest(i));
        }
        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: AutotuneKey, fastest_index: usize) {
        self.cache.insert(key, fastest_index);
    }
}
