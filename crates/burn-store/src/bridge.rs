//! Bridge between [`TensorSnapshot`] (burn-core) and the tensor-agnostic burnpack format
//! entries, used by [`BurnpackStore`](crate::BurnpackStore).
//!
//! Both directions stay lazy, and both are conversions on [`TensorSnapshot`]. Saving goes
//! through [`From<TensorSnapshot>`], which builds a deferred [`PackTensor`] holding the
//! snapshot until the writer reaches it; loading goes through [`From<PackTensor>`], which
//! leaves a reader's (possibly file-backed) bytes unread until the snapshot is materialized.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use burn_pack::{Error as PackError, Tensor as PackTensor};

use super::{TensorSnapshot, TensorSnapshotError};
use burn_core::module::ParamId;
use burn_core::tensor::TensorData;

/// Turns a snapshot into a [`PackTensor`] that materializes only when written.
///
/// [`TensorSnapshot::data_len`] derives the byte length from the snapshot's cached shape and
/// dtype, so [`Writer`](burn_pack::Writer) can lay out the whole container (descriptors,
/// offsets, total size) without calling `to_data()` on anything. Each snapshot is then
/// materialized only once the writer reaches its slot in the data section, and dropped
/// before the next one is produced.
impl From<TensorSnapshot> for PackTensor {
    fn from(snapshot: TensorSnapshot) -> Self {
        let (dtype, shape) = (snapshot.dtype, snapshot.shape.clone());
        let param_id = snapshot.tensor_id.map(|id| id.val());

        PackTensor::deferred(
            snapshot.full_path(),
            dtype,
            shape,
            param_id,
            snapshot.data_len(),
            move || {
                snapshot
                    .to_data()
                    .map(|data| data.bytes)
                    .map_err(|e| match e {
                        // Only a genuine read failure is an I/O error. Materialization also fails
                        // when the backend panics reading a tensor back from the device, and
                        // calling that an I/O error sends the reader looking at their disk: it
                        // arrives mid-write, alongside the writer's own file errors, and reads
                        // exactly like one of them.
                        TensorSnapshotError::IoError(message) => PackError::IoError(message),
                        other => PackError::ValidationError(other.to_string()),
                    })
            },
        )
    }
}

/// Turns a reader's [`PackTensor`] entry into a lazy [`TensorSnapshot`].
///
/// The counterpart to the impl above, and lazy in the same way: the tensor's
/// [`burn_pack::Bytes`] may be file-backed (from [`burn_pack::Reader::from_file`]), in which
/// case the data is only read from disk when the snapshot is materialized.
impl From<PackTensor> for TensorSnapshot {
    fn from(tensor: PackTensor) -> Self {
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let path_stack: Vec<String> = tensor.name.split('.').map(|s| s.to_string()).collect();
        let tensor_id = tensor.param_id.map(ParamId::from).unwrap_or_default();

        let data_fn = TensorSnapshot::data_fn(move || {
            // Classified as in the save direction above, in reverse: here the non-I/O cases
            // are a malformed container or a length disagreeing with its descriptor.
            let bytes = tensor.to_bytes().map_err(|e| match e {
                PackError::IoError(message) => TensorSnapshotError::IoError(message),
                other => TensorSnapshotError::DataError(other.to_string()),
            })?;
            Ok(TensorData::from_bytes(bytes, tensor.shape.clone(), dtype))
        });

        TensorSnapshot::from_closure(data_fn, dtype, shape, path_stack, vec![], tensor_id)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::string::ToString;
    // Tests only build for std targets, so the atomic pointer is always available here.
    use alloc::sync::Arc;
    use burn_core::tensor::{DType, Tensor, shape};

    #[test]
    fn deferred_tensor_carries_the_snapshot_metadata() {
        let device = Default::default();
        let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
        let id = ParamId::new();
        let snapshot = TensorSnapshot::from_float(
            &tensor,
            vec!["encoder".to_string(), "weight".to_string()],
            vec![],
            id,
        );

        let packed = PackTensor::from(snapshot);
        assert_eq!(packed.name, "encoder.weight");
        assert_eq!(packed.shape, shape![2, 2]);
        assert_eq!(packed.dtype, DType::F32);
        assert_eq!(packed.param_id, Some(id.val()));
        // That the conversion defers rather than materializes is pinned behaviourally by
        // `writing_snapshots_materializes_each_one_once_and_only_while_writing` below.
        assert_eq!(packed.byte_len(), 16);
    }

    /// A materialization failure must reach the caller with its class and its tensor intact:
    /// only a genuine read failure is an I/O error, and everything else (a backend panic, a
    /// data error) must not masquerade as one, because it arrives mid-write alongside the
    /// writer's real file errors and would send the user checking their disk. Reverting the
    /// match in `into_bytes` to a blanket `IoError`, or dropping the writer's `in_tensor`
    /// annotation, fails here.
    #[test]
    fn materialization_failures_keep_their_class_and_name() {
        use burn_pack::Error as PackError;

        let failing = |error: TensorSnapshotError| {
            TensorSnapshot::from_closure(
                Arc::new(move || Err(error.clone())),
                DType::F32,
                shape![1],
                vec!["weight".to_string()],
                vec![],
                ParamId::new(),
            )
        };

        let io = TensorSnapshotError::IoError("read failed".to_string());
        let err = burn_pack::Writer::new(vec![PackTensor::from(failing(io))])
            .into_bytes()
            .unwrap_err();
        assert!(
            matches!(&err, PackError::IoError(m) if m.contains("tensor 'weight'") && m.contains("read failed")),
            "expected a named IoError, got {err:?}"
        );

        let panic = TensorSnapshotError::PanicError("device readback panicked".to_string());
        let err = burn_pack::Writer::new(vec![PackTensor::from(failing(panic))])
            .into_bytes()
            .unwrap_err();
        assert!(
            matches!(&err, PackError::ValidationError(m) if m.contains("tensor 'weight'") && m.contains("device readback panicked")),
            "expected a named ValidationError, got {err:?}"
        );
    }

    /// The whole point of the deferred conversion is that the writer can lay out a container
    /// without materializing anything, then draw one tensor at a time. burn-pack has its own
    /// tests for that, but against a test double: this one pins the shipped path, so that
    /// wiring `byte_len` to `to_data().bytes.len()` (the obvious "fix" if `data_len` ever
    /// looks wrong) fails here instead of silently restoring the old peak memory.
    #[test]
    fn writing_snapshots_materializes_each_one_once_and_only_while_writing() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));

        let snapshots: Vec<TensorSnapshot> = (0..3)
            .map(|i| {
                let calls = calls.clone();
                let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
                TensorSnapshot::from_closure(
                    Arc::new(move || {
                        calls.fetch_add(1, Ordering::Relaxed);
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
        let size = burn_pack::Writer::new(
            snapshots
                .clone()
                .into_iter()
                .map(PackTensor::from)
                .collect(),
        )
        .size()
        .unwrap();
        assert!(size > 0);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "laying out the container materialized data"
        );

        // Writing draws from each snapshot exactly once.
        burn_pack::Writer::new(snapshots.into_iter().map(PackTensor::from).collect())
            .into_bytes()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }
}
