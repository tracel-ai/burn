//! Bridge between [`TensorSnapshot`] (burn-core) and the tensor-agnostic burnpack format
//! entries, used by [`BurnpackStore`](crate::BurnpackStore).
//!
//! Both directions are trait impls on [`TensorSnapshot`], and both stay lazy. Saving goes
//! through [`TensorEntry`], which defers each snapshot until the writer reaches it; loading
//! goes through `From<PackTensor>`, which leaves a reader's (possibly file-backed) bytes
//! unread until the snapshot is materialized.

use alloc::borrow::Cow;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_pack::{Bytes, DType, Error as PackError, Shape, Tensor as PackTensor, TensorEntry};

use super::{TensorSnapshot, TensorSnapshotError};
use burn_core::module::ParamId;
use burn_core::tensor::TensorData;

/// Lets a [`Writer`](burn_pack::Writer) consume snapshots without materializing them.
///
/// [`TensorSnapshot::data_len`] derives the byte length from the snapshot's cached shape and
/// dtype, so the writer can lay out the whole container (descriptors, offsets, total size)
/// without calling `to_data()` on anything. Each snapshot is then materialized only once the
/// writer reaches its slot in the data section, and dropped before the next one is produced.
impl TensorEntry for TensorSnapshot {
    fn name(&self) -> Cow<'_, str> {
        // Owned: the full path is joined from the path stack, so there is nothing to borrow.
        Cow::Owned(self.full_path())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn param_id(&self) -> Option<u64> {
        self.tensor_id.map(|id| id.val())
    }

    fn byte_len(&self) -> usize {
        self.data_len()
    }

    fn into_bytes(self) -> Result<Bytes, PackError> {
        self.to_data().map(|data| data.bytes).map_err(|e| match e {
            // Only a genuine read failure is an I/O error. Materialization also fails when the
            // backend panics reading a tensor back from the device, and calling that an I/O
            // error sends the reader looking at their disk: it arrives mid-write, alongside
            // the writer's own file errors, and reads exactly like one of them.
            TensorSnapshotError::IoError(message) => PackError::IoError(message),
            other => PackError::ValidationError(other.to_string()),
        })
    }
}

/// Turns a reader's [`PackTensor`] entry into a lazy [`TensorSnapshot`].
///
/// The counterpart to the [`TensorEntry`] impl above, and lazy in the same way: the tensor's
/// [`burn_pack::Bytes`] may be file-backed (from [`burn_pack::Reader::from_file`]), in which
/// case the data is only read from disk when the snapshot is materialized.
impl From<PackTensor> for TensorSnapshot {
    fn from(tensor: PackTensor) -> Self {
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let path_stack: Vec<String> = tensor.name.split('.').map(|s| s.to_string()).collect();
        let tensor_id = tensor.param_id.map(ParamId::from).unwrap_or_default();

        let bytes = tensor.bytes;
        let shape_for_closure = shape.clone();
        let data_fn = Rc::new(move || {
            Ok(TensorData::from_bytes(
                bytes.clone(),
                shape_for_closure.clone(),
                dtype,
            ))
        });

        TensorSnapshot::from_closure(data_fn, dtype, shape, path_stack, vec![], tensor_id)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use burn_core::tensor::{Tensor, shape};

    #[test]
    fn entry_exposes_snapshot_metadata() {
        let device = Default::default();
        let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let id = ParamId::new();
        let snapshot = TensorSnapshot::from_float(
            &tensor,
            vec!["encoder".to_string(), "weight".to_string()],
            vec![],
            id,
        );

        assert_eq!(TensorEntry::name(&snapshot), "encoder.weight");
        assert_eq!(TensorEntry::shape(&snapshot), &shape![2, 2]);
        assert_eq!(TensorEntry::dtype(&snapshot), DType::F32);
        assert_eq!(TensorEntry::byte_len(&snapshot), 16);
        assert_eq!(snapshot.param_id(), Some(id.val()));
    }

    /// The whole point of the [`TensorEntry`] impl is that the writer can lay out a container
    /// without materializing anything, then draw one tensor at a time. burn-pack has its own
    /// tests for that, but against a test double: this one pins the shipped path, so that
    /// wiring `byte_len` to `to_data().bytes.len()` (the obvious "fix" if `data_len` ever
    /// looks wrong) fails here instead of silently restoring the old peak memory.
    #[test]
    fn writing_snapshots_materializes_each_one_once_and_only_while_writing() {
        let calls = Rc::new(core::cell::Cell::new(0usize));

        let snapshots: Vec<TensorSnapshot> = (0..3)
            .map(|i| {
                let calls = calls.clone();
                let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
                TensorSnapshot::from_closure(
                    Rc::new(move || {
                        calls.set(calls.get() + 1);
                        Ok(data.clone())
                    }),
                    DType::F32,
                    shape![4],
                    vec![format!("layer{i}"), "weight".to_string()],
                    vec![],
                    ParamId::new(),
                )
            })
            .collect();

        // Planning reads only the cached metadata.
        let size = burn_pack::Writer::new(snapshots.clone()).size().unwrap();
        assert!(size > 0);
        assert_eq!(calls.get(), 0, "laying out the container materialized data");

        // Writing draws from each snapshot exactly once.
        burn_pack::Writer::new(snapshots).into_bytes().unwrap();
        assert_eq!(calls.get(), 3);
    }
}
