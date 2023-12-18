use std::fs;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use super::AutotuneKey;
use super::AutotuneOperation;
use super::AutotuneOperationSet;
use alloc::boxed::Box;
use hashbrown::HashMap;

/// Use to find and reuse the best kernel for some input
#[derive(Debug, Default)]
pub(crate) struct TuneCache<K> {
    cache: HashMap<K, (bool, usize)>,
    persistent_cache: HashMap<String, (String, usize)>,
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
            persistent_cache: HashMap::new(),
        }
    }

    #[allow(clippy::borrowed_box)]
    pub(crate) fn try_cache(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
    ) -> TuneCacheResult<K> {
        let key = autotune_operation_set.key();
        let result = self.cache.get_mut(&key);
        println!("result {:?}", result);
        if let Some((is_checked, index)) = result {
            if !*is_checked {
                let checksum = autotune_operation_set.compute_checksum();
                let (expected_checksum, _) = self
                    .persistent_cache
                    .get(&key.to_string())
                    .expect("Both caches should be in sync");
                println!("{} -- {}", checksum, expected_checksum);
                if &checksum != expected_checksum {
                    return TuneCacheResult::Miss(autotune_operation_set);
                }
                *is_checked = true;
            }
            return TuneCacheResult::Hit(autotune_operation_set.fastest(*index));
        }
        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: K, fastest_index: usize, checksum: String) {
        self.persistent_cache.insert(key.to_string(), (checksum, fastest_index));
        self.cache.insert(key, (true, fastest_index));
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) -> Result<(), io::Error> {
        let file_path  = TuneCache::<K>::get_persistent_cache_file_path();
        // note: reading file ro memory is faster than using
        // serde from_reader with a buffered reader
        // see issue:
        // https://github.com/serde-rs/json/issues/160
        let data = fs::read_to_string(file_path)?;
        self.persistent_cache = serde_json::from_str(&data)?;
        dbg!(&self.persistent_cache);
        Ok(())
    }

    /// Save the persistent cache on disk
    pub(crate) fn save(&self) {
        let file_path  = TuneCache::<K>::get_persistent_cache_file_path();
        let file = File::create(&file_path).expect(
            "Unable to open autotune persistent cache file");
        serde_json::to_writer_pretty(file, &self.persistent_cache)
            .expect("Unable to write to autotune persistent cache");
    }

    /// Return the file path for the persistent cache on disk
    fn get_persistent_cache_file_path() -> PathBuf {
        let home_dir = dirs::home_dir().expect("Could not get home directory");
        let path_dir = home_dir.join(".cache").join("burn").join("autotune");
        let path = Path::new(&path_dir);
        let file_path = path.join("autotune-cache.json");
        file_path
    }
}
