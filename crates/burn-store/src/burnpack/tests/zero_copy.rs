//! Tests for zero-copy tensor loading functionality.

use crate::ModuleStore;
use crate::burnpack::store::BurnpackStore;

use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_tensor::{AllocationProperty, Bytes, Tensor, backend::Backend};

type TestBackend = burn_ndarray::NdArray;

#[derive(Module, Debug)]
struct SimpleModule<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> SimpleModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::from_data([[1.0f32, 2.0], [3.0, 4.0]], device),
            bias: Param::from_data([0.5f32, 1.5], device),
        }
    }

    fn new_zeros(device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([2, 2], device)),
            bias: Param::from_tensor(Tensor::zeros([2], device)),
        }
    }
}

/// Test that from_static creates a store with zero_copy enabled by default.
#[test]
fn test_from_static_enables_zero_copy() {
    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to bytes first
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Convert to Vec<u8> and then leak to get &'static [u8]
    let bytes_vec: Vec<u8> = bytes.to_vec();
    let static_bytes: &'static [u8] = Box::leak(bytes_vec.into_boxed_slice());

    // Create store from static - zero_copy should be enabled
    let mut load_store = BurnpackStore::from_static(static_bytes);

    // Load into a new module
    let mut loaded_module = SimpleModule::<TestBackend>::new_zeros(&device);
    load_store.apply_to(&mut loaded_module).unwrap();

    // Verify data is correct
    let loaded_weight = loaded_module.weight.val().to_data();
    let loaded_bias = loaded_module.bias.val().to_data();

    assert_eq!(
        loaded_weight.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(loaded_bias.to_vec::<f32>().unwrap(), vec![0.5, 1.5]);
}

/// Test that zero_copy builder method works.
#[test]
fn test_zero_copy_builder_method() {
    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to bytes first
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Create shared bytes for zero-copy
    let shared = bytes::Bytes::from(bytes.to_vec());
    let cubecl_bytes = Bytes::from_shared(shared, AllocationProperty::Other);

    // Create store with zero_copy enabled
    let mut load_store = BurnpackStore::from_bytes(Some(cubecl_bytes)).zero_copy(true);

    // Load into a new module
    let mut loaded_module = SimpleModule::<TestBackend>::new_zeros(&device);
    load_store.apply_to(&mut loaded_module).unwrap();

    // Verify data is correct
    let loaded_weight = loaded_module.weight.val().to_data();
    assert_eq!(
        loaded_weight.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// Test that zero_copy(false) uses copying even with shared bytes.
#[test]
fn test_zero_copy_disabled_uses_copy() {
    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to bytes first
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Convert to Vec<u8> and then leak to get &'static [u8]
    let bytes_vec: Vec<u8> = bytes.to_vec();
    let static_bytes: &'static [u8] = Box::leak(bytes_vec.into_boxed_slice());

    // Create store from static but disable zero_copy
    let mut load_store = BurnpackStore::from_static(static_bytes).zero_copy(false);

    // Load into a new module
    let mut loaded_module = SimpleModule::<TestBackend>::new_zeros(&device);
    load_store.apply_to(&mut loaded_module).unwrap();

    // Verify data is correct (copied, not zero-copy)
    let loaded_weight = loaded_module.weight.val().to_data();
    assert_eq!(
        loaded_weight.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// Test that from_bytes with regular Bytes uses copying by default.
#[test]
fn test_from_bytes_uses_copy_by_default() {
    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load from bytes (default: zero_copy = false)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut loaded_module = SimpleModule::<TestBackend>::new_zeros(&device);
    load_store.apply_to(&mut loaded_module).unwrap();

    // Verify data is correct
    let loaded_weight = loaded_module.weight.val().to_data();
    assert_eq!(
        loaded_weight.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// Test that slice_bytes works correctly on StorageBackend.
#[test]
fn test_storage_backend_slice_bytes() {
    use crate::burnpack::reader::BurnpackReader;

    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to bytes first
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Create shared bytes
    let shared = bytes::Bytes::from(bytes.to_vec());
    let cubecl_bytes = Bytes::from_shared(shared, AllocationProperty::Other);

    // Create reader and get snapshots with zero-copy
    let reader = BurnpackReader::from_bytes(cubecl_bytes).unwrap();
    let snapshots = reader.get_snapshots_zero_copy(true).unwrap();

    // Verify we got the expected number of tensors
    assert_eq!(snapshots.len(), 2);

    // Load the tensor data
    for snapshot in &snapshots {
        let data = snapshot.to_data().unwrap();
        // Just verify we can access the data - the actual content depends on tensor order
        assert!(!data.bytes.is_empty());
    }
}

/// Test that zero_copy=true with file-based loading works (via mmap + bytes::Bytes).
#[test]
fn test_zero_copy_file_based_works() {
    use tempfile::NamedTempFile;

    let device = Default::default();
    let module = SimpleModule::<TestBackend>::new(&device);

    // Save to a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    let mut save_store = BurnpackStore::from_file(path).overwrite(true);
    save_store.collect_from(&module).unwrap();

    // Load with zero_copy=true - should work because mmap is converted to bytes::Bytes
    let mut load_store = BurnpackStore::from_file(path).zero_copy(true);
    let mut loaded_module = SimpleModule::<TestBackend>::new_zeros(&device);

    // The apply should succeed - mmap now supports zero-copy via bytes::Bytes::from_owner()
    load_store.apply_to(&mut loaded_module).unwrap();

    // Verify data is correct
    let loaded_weight = loaded_module.weight.val().to_data();
    assert_eq!(
        loaded_weight.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}
