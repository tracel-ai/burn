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
#[derive(Debug)]
pub(crate) struct TuneCache<K> {
    cache: HashMap<K, (bool, usize)>,
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
        let mut cache = TuneCache {
            cache: HashMap::new(),
            persistent_cache: HashMap::new(),
        };
        if let Err(e) = cache.load() {
            eprintln!(
                "Unable to load autotune cache. Cache will be ignored ({}).", e);
        }
        cache
    }

    #[allow(clippy::borrowed_box)]
    pub(crate) fn try_cache(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K>>,
    ) -> TuneCacheResult<K> {
        let key = autotune_operation_set.key();
        let result = self.cache.get_mut(&key);
        if let Some((is_checked, index)) = result {
            if !*is_checked {
                let checksum = autotune_operation_set.compute_checksum();
                let (expected_checksum, _) = self
                    .persistent_cache
                    .get(&key)
                    .expect("Both caches should be in sync");
                println!("{}: {} -- {}", key, checksum, expected_checksum);
                if &checksum != expected_checksum {
                    return TuneCacheResult::Miss(autotune_operation_set);
                }
                *is_checked = true;
            }
            return TuneCacheResult::Hit(autotune_operation_set.fastest(*index));
        }
        TuneCacheResult::Miss(autotune_operation_set)
    }

    pub(crate) fn cache_insert(&mut self, key: K, checksum: String, fastest_index: usize) {
        println!("Inserting key: {}", key);
        self.persistent_cache.insert(key.clone(), (checksum, fastest_index));
        self.cache.insert(key, (true, fastest_index));
    }

    /// Load the persistent cache data from disk
    pub(crate) fn load(&mut self) -> Result<(), io::Error> {
        let file_path  = TuneCache::<K>::get_persistent_cache_file_path();
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
            },
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
        };
        Ok(())
    }

    /// Save the persistent cache on disk
    pub(crate) fn save(&self) {
        let file_path  = TuneCache::<K>::get_persistent_cache_file_path();
        let file = File::create(&file_path).expect(
            "Unable to open autotune persistent cache file");
        let data = self.persistent_cache.iter().collect::<Vec<_>>();
        serde_json::to_writer_pretty(file, &data)
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

