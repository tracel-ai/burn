#[cfg(feature = "autotune-persistent-cache")]
mod std_imports {
    pub use std::fs;
    pub use std::fs::File;
    pub use std::io;
    pub use std::path::Path;
    pub use std::path::PathBuf;
}

#[cfg(feature = "autotune-persistent-cache")]
use std_imports::*;

use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;
use hashbrown::HashMap;

/// Return the file path for the persistent cache on disk
#[cfg(feature = "autotune-persistent-cache")]
pub fn get_persistent_cache_file_path() -> PathBuf {
    let home_dir = dirs::home_dir().expect("An home directory should exist");
    let path_dir = home_dir.join(".cache").join("burn").join("autotune");
    let path = Path::new(&path_dir);
    path.join("autotune-cache.json")
}

/// Use to find and reuse the best kernel for some input
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    cache: HashMap<K, (bool, usize)>,
    #[cfg(feature = "autotune-persistent-cache")]
    persistent_cache: HashMap<K, (String, usize)>,
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
        #[cfg(feature = "autotune-persistent-cache")]
        {
            let mut cache = TuneCache {
                cache: HashMap::new(),
                persistent_cache: HashMap::new(),
            };
            if let Err(e) = cache.load() {
                log::warn!(
                    "Unable to load autotune cache. Cache will be ignored ({}).",
                    e
                );
            }
            cache
        }

        #[cfg(not(feature = "autotune-persistent-cache"))]
        {
            TuneCache {
                cache: HashMap::new(),
            }
        }
    }

    pub(crate) fn try_cache(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
    ) -> TuneCacheResult<K> {
        let key = autotune_operation_set.key();
        let result = self.cache.get_mut(&key);

        #[cfg(feature = "autotune-persistent-cache")]
        {
            if let Some((is_checked, index)) = result {
                if !*is_checked {
                    let checksum = autotune_operation_set.compute_checksum();
                    let (expected_checksum, _) = self
                        .persistent_cache
                        .get(&key)
                        .expect("Both caches should be in sync");
                    if &checksum != expected_checksum {
                        return TuneCacheResult::Miss(autotune_operation_set);
                    }
                    *is_checked = true;
                }
                return TuneCacheResult::Hit(autotune_operation_set.fastest(*index));
            }
        }

        #[cfg(not(feature = "autotune-persistent-cache"))]
        {
            if let Some((_is_checked, index)) = result {
                return TuneCacheResult::Hit(autotune_operation_set.fastest(*index));
            }
        }

        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize) {
        self.cache.insert(key, (true, fastest_index));
    }

    #[cfg(feature = "autotune-persistent-cache")]
    pub(crate) fn persistent_cache_insert(
        &mut self,
        key: K,
        checksum: String,
        fastest_index: usize,
    ) {
        self.persistent_cache.insert(key, (checksum, fastest_index));
    }

    /// Load the persistent cache data from disk
    #[cfg(feature = "autotune-persistent-cache")]
    pub(crate) fn load(&mut self) -> Result<(), io::Error> {
        let file_path = get_persistent_cache_file_path();
        // note: reading file ro memory is faster than using
        // serde from_reader with a buffered reader
        // see issue:
        // https://github.com/serde-rs/json/issues/160
        match fs::read_to_string(file_path) {
            Ok(data) => {
                let data: Vec<(K, (String, usize))> = serde_json::from_str(&data)?;
                for (key, value) in data.into_iter() {
                    self.persistent_cache.insert(key, value);
                }
                Ok(())
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }?;
        for (key, (_checksum, index)) in self.persistent_cache.iter() {
            self.cache.insert(key.clone(), (false, *index));
        }
        Ok(())
    }

    /// Save the persistent cache on disk
    #[cfg(feature = "autotune-persistent-cache")]
    pub(crate) fn save(&self) {
        let file_path = get_persistent_cache_file_path();
        let expect_msg = format!(
            "Should be able to open autotune persistent cache file: {:?}",
            &file_path.to_str().unwrap()
        );
        let file = File::create(file_path).expect(&expect_msg);
        let data = self.persistent_cache.iter().collect::<Vec<_>>();
        serde_json::to_writer_pretty(file, &data)
            .expect("Should be able to write to autotune persistent cache");
    }
}
