use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;
use hashbrown::HashMap;

/// Use to find and reuse the best kernel for some input
#[derive(Debug, Default)]
pub(crate) struct TuneCache<K> {
    cache: HashMap<K, usize>,
}

/// Result of the cache try
pub enum TuneCacheResult<K> {
    /// An operation is found and given
    Hit(Box<dyn AutotuneOperation>),
    /// No operation is found and the set is given back for ownership
    Miss(Box<dyn AutotuneOperationSet<K>>),
}

impl<K: AutotuneKey> TuneCache<K> {
    pub(crate) fn new() -> Self {
        TuneCache {
            cache: HashMap::new(),
        }
    }

    #[allow(clippy::borrowed_box)]
    pub(crate) fn try_cache(
        &self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
    ) -> TuneCacheResult<K> {
        let index = self.cache.get(&autotune_operation_set.key());
        if let Some(&i) = index {
            return TuneCacheResult::Hit(autotune_operation_set.fastest(i));
        }
        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.cache.insert(key, fastest_index);
    }
}
