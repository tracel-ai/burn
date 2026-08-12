//! Bridge between [`TensorSnapshot`] (burn-core) and the tensor-agnostic burnpack format
//! entries, used by [`BurnpackStore`](crate::BurnpackStore).
//!
//! Saving goes through the [`TensorEntry`] impl below, which keeps each snapshot lazy until
//! the writer reaches it. Loading goes through [`tensor_to_snapshot`], which keeps a
//! reader's (possibly file-backed) bytes lazy until the snapshot is materialized.

use alloc::borrow::Cow;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_pack::{Bytes, DType, Error as PackError, Shape, Tensor as PackTensor, TensorEntry};

use super::TensorSnapshot;
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
        self.to_data()
            .map(|data| data.bytes)
            .map_err(|e| PackError::IoError(format!("{e:?}")))
    }
}

/// Convert a [`PackTensor`] entry into a lazy [`TensorSnapshot`].
///
/// The tensor's [`Bytes`](burn_pack::Bytes) may be file-backed (from [`Reader::from_file`](burn_pack::Reader::from_file)),
/// in which case the data is only read from disk when the snapshot is materialized.
pub fn tensor_to_snapshot(tensor: PackTensor) -> TensorSnapshot {
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
        assert_eq!(snapshot.param_id(), Some(id.val()));
    }
}
