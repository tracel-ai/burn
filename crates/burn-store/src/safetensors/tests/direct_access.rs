use burn_core as burn;

use crate::{ModuleStore, SafetensorsStore};
use burn_core::module::{Module, Param};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

type TestBackend = burn_ndarray::NdArray;

// Test module for direct access tests
#[derive(Module, Debug)]
struct DirectAccessTestModule<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
    nested: DirectAccessNestedModule<B>,
}

#[derive(Module, Debug)]
struct DirectAccessNestedModule<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> DirectAccessTestModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
            bias: Param::from_data([0.1, 0.2], device),
            nested: DirectAccessNestedModule {
                gamma: Param::from_data([1.0, 2.0], device),
                beta: Param::from_data([0.5, 0.5], device),
            },
        }
    }
}

// ============================================================================
// Tests for MemoryStore variant
// ============================================================================

#[test]
fn test_memory_get_all_snapshots() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    // Save module to memory
    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();

    // Get bytes and create load store
    let bytes = save_store.get_bytes().unwrap();
    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    // Get all snapshots
    let snapshots = load_store.get_all_snapshots().unwrap();

    assert_eq!(snapshots.len(), 4);
    assert!(snapshots.contains_key("weight"));
    assert!(snapshots.contains_key("bias"));
    assert!(snapshots.contains_key("nested.gamma"));
    assert!(snapshots.contains_key("nested.beta"));
}

#[test]
fn test_memory_get_snapshot_existing() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    // Get existing snapshot
    let snapshot = load_store.get_snapshot("weight").unwrap();
    assert!(snapshot.is_some());

    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.shape, vec![2, 2]);

    // Verify data
    let data = snapshot.to_data().unwrap();
    let values: Vec<f32> = data.to_vec().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_memory_get_snapshot_nested() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    // Get nested snapshot
    let snapshot = load_store.get_snapshot("nested.gamma").unwrap();
    assert!(snapshot.is_some());

    let snapshot = snapshot.unwrap();
    let data = snapshot.to_data().unwrap();
    let values: Vec<f32> = data.to_vec().unwrap();
    assert_eq!(values, vec![1.0, 2.0]);
}

#[test]
fn test_memory_get_snapshot_not_found() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    // Get non-existent snapshot
    let snapshot = load_store.get_snapshot("nonexistent").unwrap();
    assert!(snapshot.is_none());
}

#[test]
fn test_memory_keys() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    let keys = load_store.keys().unwrap();
    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"weight".to_string()));
    assert!(keys.contains(&"bias".to_string()));
    assert!(keys.contains(&"nested.gamma".to_string()));
    assert!(keys.contains(&"nested.beta".to_string()));
}

#[test]
fn test_memory_caching_behavior() {
    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let mut save_store = SafetensorsStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));

    // Call get_all_snapshots multiple times - should return same cached data
    let snapshots1 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots1.len(), 4);

    let snapshots2 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots2.len(), 4);

    // Verify we can still access individual snapshots after caching
    let snapshot = load_store.get_snapshot("bias").unwrap();
    assert!(snapshot.is_some());
}

// ============================================================================
// Tests for FileStore variant
// ============================================================================

#[test]
#[cfg(feature = "std")]
fn test_file_get_all_snapshots() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_get_all_snapshots.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    let mut load_store = SafetensorsStore::from_file(&path);
    let snapshots = load_store.get_all_snapshots().unwrap();

    assert_eq!(snapshots.len(), 4);
    assert!(snapshots.contains_key("weight"));
    assert!(snapshots.contains_key("bias"));
    assert!(snapshots.contains_key("nested.gamma"));
    assert!(snapshots.contains_key("nested.beta"));
}

#[test]
#[cfg(feature = "std")]
fn test_file_get_snapshot_existing() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_get_snapshot.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    let mut load_store = SafetensorsStore::from_file(&path);

    let snapshot = load_store.get_snapshot("weight").unwrap();
    assert!(snapshot.is_some());

    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.shape, vec![2, 2]);

    let data = snapshot.to_data().unwrap();
    let values: Vec<f32> = data.to_vec().unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[cfg(feature = "std")]
fn test_file_get_snapshot_not_found() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_not_found.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    let mut load_store = SafetensorsStore::from_file(&path);

    let snapshot = load_store.get_snapshot("nonexistent").unwrap();
    assert!(snapshot.is_none());
}

#[test]
#[cfg(feature = "std")]
fn test_file_keys() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_keys.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    let mut load_store = SafetensorsStore::from_file(&path);

    let keys = load_store.keys().unwrap();
    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"weight".to_string()));
    assert!(keys.contains(&"bias".to_string()));
    assert!(keys.contains(&"nested.gamma".to_string()));
    assert!(keys.contains(&"nested.beta".to_string()));
}

#[test]
#[cfg(feature = "std")]
fn test_file_keys_fast_path() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_keys_fast.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Create fresh store - cache should be empty
    let mut load_store = SafetensorsStore::from_file(&path);

    // keys() should work without populating the full cache (fast path)
    let keys = load_store.keys().unwrap();
    assert_eq!(keys.len(), 4);

    // Now call get_all_snapshots to populate cache
    let snapshots = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots.len(), 4);

    // keys() should now use the cached data
    let keys2 = load_store.keys().unwrap();
    assert_eq!(keys2.len(), 4);
}

#[test]
#[cfg(feature = "std")]
fn test_file_caching_behavior() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_caching.safetensors");

    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    let mut load_store = SafetensorsStore::from_file(&path);

    // First call populates cache
    let snapshots1 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots1.len(), 4);

    // Second call uses cache
    let snapshots2 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots2.len(), 4);
}

#[test]
#[cfg(feature = "std")]
fn test_file_cache_invalidation_on_save() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = DirectAccessTestModule::<TestBackend>::new(&device);

    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_invalidation.safetensors");

    // Create store, save, and populate cache
    let mut store = SafetensorsStore::from_file(&path).overwrite(true);
    store.collect_from(&module).unwrap();

    let snapshots1 = store.get_all_snapshots().unwrap();
    assert_eq!(snapshots1.len(), 4);

    // Save again (this should invalidate cache)
    store.collect_from(&module).unwrap();

    // Cache should be repopulated with fresh data
    let snapshots2 = store.get_all_snapshots().unwrap();
    assert_eq!(snapshots2.len(), 4);
}
