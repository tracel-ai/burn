//! Bridge between [`TensorSnapshot`] (burn-core) and [`burn_store::BurnpackTensor`]
//! (the tensor-agnostic burnpack format entry).
//!
//! These conversions live in burn-core so the burnpack format crate (`burn-store`)
//! stays free of any tensor dependency. They are public so that higher layers (e.g.
//! `burn-import`'s native store) can serialize/deserialize snapshots through the
//! burnpack format.

use alloc::format;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_store::{BurnpackError, BurnpackTensor};

use super::{TensorSnapshot, TensorSnapshotError};
use crate::module::ParamId;
use crate::tensor::{Shape, TensorData};

/// Convert a lazy [`TensorSnapshot`] into a lazy [`BurnpackTensor`] entry.
pub fn snapshot_to_tensor(snapshot: &TensorSnapshot) -> BurnpackTensor {
    let data_fn = snapshot.clone_data_fn();
    let shape: Vec<usize> = snapshot.shape.iter().copied().collect();
    BurnpackTensor::new(
        snapshot.full_path(),
        snapshot.dtype,
        shape,
        snapshot.tensor_id.map(|id| id.val()),
        snapshot.data_len(),
        Rc::new(move || {
            data_fn()
                .map(|data| data.bytes)
                .map_err(|e| BurnpackError::IoError(format!("{e:?}")))
        }),
    )
}

/// Convert a lazy [`BurnpackTensor`] entry back into a lazy [`TensorSnapshot`].
pub fn tensor_to_snapshot(tensor: BurnpackTensor) -> TensorSnapshot {
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
