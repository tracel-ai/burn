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

pub enum TuneCacheResult<S> {
    Hit(Box<dyn AutotuneOperation<S>>),
    Miss(Box<dyn AutotuneOperationSet<S>>),
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
        autotune_operation_set: Box<dyn AutotuneOperationSet<S>>,
    ) -> TuneCacheResult<S> {
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
