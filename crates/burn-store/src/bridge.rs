//! Bridge between [`TensorSnapshot`] (burn-core) and [`burn_pack::Tensor`]
//! (the tensor-agnostic burnpack format entry), used by [`BurnpackStore`](crate::BurnpackStore).

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_pack::{Error as PackError, Tensor as PackTensor};

use super::TensorSnapshot;
use burn_core::module::ParamId;
use burn_core::tensor::TensorData;

/// Convert a [`TensorSnapshot`] into a [`PackTensor`] entry, materializing its data.
pub fn snapshot_to_tensor(snapshot: &TensorSnapshot) -> Result<PackTensor, PackError> {
    let data = snapshot
        .to_data()
        .map_err(|e| PackError::IoError(format!("{e:?}")))?;
    Ok(PackTensor::new(
        snapshot.full_path(),
        snapshot.dtype,
        snapshot.shape.clone(),
        snapshot.tensor_id.map(|id| id.val()),
        data.bytes,
    ))
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
