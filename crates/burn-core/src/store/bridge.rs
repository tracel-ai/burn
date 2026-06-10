//! Bridge between [`TensorSnapshot`] (burn-core) and [`burn_pack::Tensor`]
//! (the tensor-agnostic burnpack format entry).
//!
//! These conversions live in burn-core so the burnpack format crate (`burn-pack`)
//! stays free of any tensor dependency. They are public so that higher layers (e.g.
//! `burn-store`'s native store) can serialize/deserialize snapshots through the
//! burnpack format.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_pack::{Error as PackError, Tensor as PackTensor};

use super::{TensorSnapshot, TensorSnapshotError};
use crate::module::ParamId;
use crate::tensor::{Shape, TensorData};

/// Convert a lazy [`TensorSnapshot`] into a lazy [`PackTensor`] entry.
pub fn snapshot_to_tensor(snapshot: &TensorSnapshot) -> PackTensor {
    let data_fn = snapshot.clone_data_fn();
    let shape: Vec<usize> = snapshot.shape.iter().copied().collect();
    PackTensor::new(
        snapshot.full_path(),
        snapshot.dtype,
        shape,
        snapshot.tensor_id.map(|id| id.val()),
        snapshot.data_len(),
        Rc::new(move || {
            data_fn()
                .map(|data| data.bytes)
                .map_err(|e| PackError::IoError(format!("{e:?}")))
        }),
    )
}

/// Convert a lazy [`PackTensor`] entry back into a lazy [`TensorSnapshot`].
pub fn tensor_to_snapshot(tensor: PackTensor) -> TensorSnapshot {
    let dtype = tensor.dtype;
    let shape = Shape::from(tensor.shape.clone());
    let path_stack: Vec<String> = tensor.name.split('.').map(|s| s.to_string()).collect();
    let tensor_id = tensor
        .param_id
        .map(ParamId::from)
        .unwrap_or_else(ParamId::new);

    let shape_for_closure = shape.clone();
    let data_fn = Rc::new(move || {
        let bytes = tensor
            .bytes()
            .map_err(|e| TensorSnapshotError::IoError(format!("{e}")))?;
        Ok(TensorData::from_bytes(
            bytes,
            shape_for_closure.clone(),
            dtype,
        ))
    });

    TensorSnapshot::from_closure(data_fn, dtype, shape, path_stack, vec![], tensor_id)
}
