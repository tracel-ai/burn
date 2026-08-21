//! Bridge between burn-core tensors and the tensor-agnostic burnpack format entries.
//!
//! [`burn_pack::Tensor`] carries the format-level facts about a tensor (name, dtype, shape,
//! optional param id) and a source for its raw little-endian bytes, which may not exist yet.
//! burn-store uses it as its single tensor-transport type: what a [`Collector`](crate::Collector)
//! produces, what an [`Applier`](crate::Applier) consumes, and what the stores read and write.
//!
//! This module is the seam where burn-core's [`TensorData`] meets those raw bytes. Everything
//! here stays lazy: a tensor built by [`from_tensor`] holds the (reference-counted) device
//! tensor and reads it back only when the bytes are finally drawn, and [`map_data`] wraps one
//! deferred source in another without materializing either.

use alloc::string::String;
// Only the `std` panic guard below builds an error message.
#[cfg(feature = "std")]
use alloc::string::ToString;

use burn_pack::{Error as PackError, Tensor as PackTensor};

use burn_core::tensor::kind::Basic;
use burn_core::tensor::quantization::quantized_data_len;
use burn_core::tensor::{DType, Shape, Tensor, TensorData};

/// Number of bytes a tensor of this dtype and shape serializes to.
///
/// [`Writer`](burn_pack::Writer) lays out the whole container from these lengths before any
/// data is produced, which is what lets a save's peak memory be bounded by the largest single
/// tensor rather than by the model. Quantized data packs its values and appends scales inline,
/// so its length is not a product of the shape and dtype and gets its own accounting.
pub fn data_len(dtype: DType, shape: &Shape) -> usize {
    match dtype {
        DType::QFloat(scheme) => quantized_data_len(&scheme, shape),
        _ => shape.iter().product::<usize>() * dtype.size(),
    }
}

/// Run a data-producing closure, turning a panic into an error.
///
/// A backend can panic reading a tensor back from its device, and that panic would otherwise
/// unwind through [`Writer`](burn_pack::Writer) partway through a container. burn-pack cannot
/// guard against it: `catch_unwind` needs `std`, which that crate deliberately does not
/// require. So the guard lives here, wrapped around every provider burn-store builds.
///
/// Classified as a validation failure rather than an I/O one on purpose. It arrives mid-write,
/// alongside the writer's own file errors, and calling it I/O would send the reader looking at
/// their disk.
#[cfg(feature = "std")]
fn guarded(f: impl Fn() -> Result<TensorData, PackError>) -> Result<TensorData, PackError> {
    // AssertUnwindSafe: the shared closure is not UnwindSafe, and the only state that could be
    // observed after a panic is the tensor being read, which is dropped either way.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| {
        Err(PackError::ValidationError(
            "panic while producing tensor data".to_string(),
        ))
    })
}

/// Run a data-producing closure.
///
/// See the `std` variant. `catch_unwind` is unavailable here, so a panic propagates.
#[cfg(not(feature = "std"))]
fn guarded(f: impl Fn() -> Result<TensorData, PackError>) -> Result<TensorData, PackError> {
    f()
}

/// Build a tensor whose data is produced on demand by `data_fn`.
///
/// `dtype` and `shape` describe what `data_fn` will return, and fix the byte length the writer
/// reserves, so a transform that changes either must be declared here rather than left for the
/// closure to reveal.
#[cfg(target_has_atomic = "ptr")]
pub fn deferred(
    name: String,
    dtype: DType,
    shape: Shape,
    param_id: Option<u64>,
    data_fn: impl Fn() -> Result<TensorData, PackError> + Send + Sync + 'static,
) -> PackTensor {
    let byte_len = data_len(dtype, &shape);
    PackTensor::deferred(name, dtype, shape, param_id, byte_len, move || {
        guarded(&data_fn).map(|data| data.bytes)
    })
}

/// Build a tensor whose data is produced on demand by `data_fn`.
///
/// See the `target_has_atomic = "ptr"` variant. This one drops the `Send + Sync` bound, which
/// nothing on a single-threaded target can satisfy or needs.
#[cfg(not(target_has_atomic = "ptr"))]
pub fn deferred(
    name: String,
    dtype: DType,
    shape: Shape,
    param_id: Option<u64>,
    data_fn: impl Fn() -> Result<TensorData, PackError> + 'static,
) -> PackTensor {
    let byte_len = data_len(dtype, &shape);
    PackTensor::deferred(name, dtype, shape, param_id, byte_len, move || {
        guarded(&data_fn).map(|data| data.bytes)
    })
}

/// Snapshot a module parameter without reading it back from its device.
///
/// The tensor is cloned (cheap, reference-counted) and read only when the bytes are drawn, so
/// a whole module can be traversed for its metadata alone. One function covers float, int and
/// bool parameters: [`Basic`] is what supplies `dtype`, `shape` and `to_data` for all three.
pub fn from_tensor<const D: usize, K: Basic + 'static>(
    tensor: &Tensor<D, K>,
    name: String,
    param_id: Option<u64>,
) -> PackTensor {
    let (dtype, shape) = (tensor.dtype(), tensor.shape());
    let tensor = tensor.clone();

    deferred(name, dtype, shape, param_id, move || Ok(tensor.to_data()))
}

/// Build a tensor from data already in hand.
pub fn from_data(data: TensorData, name: String, param_id: Option<u64>) -> PackTensor {
    PackTensor::new(name, data.dtype, data.shape.clone(), param_id, data.bytes)
}

/// Read a tensor's bytes back as [`TensorData`], leaving it intact.
///
/// Costs whatever the source costs: a deferred tensor re-runs its provider, since nothing is
/// cached. Prefer [`into_data`] where the tensor is no longer needed.
pub fn to_data(tensor: &PackTensor) -> Result<TensorData, PackError> {
    Ok(TensorData::from_bytes(
        tensor.to_bytes()?,
        tensor.shape.clone(),
        tensor.dtype,
    ))
}

/// Take a tensor's bytes as [`TensorData`], producing them if deferred.
pub fn into_data(tensor: PackTensor) -> Result<TensorData, PackError> {
    let (_, dtype, shape, _, bytes) = tensor.into_parts()?;
    Ok(TensorData::from_bytes(bytes, shape, dtype))
}

/// Wrap a tensor's data in a transform, keeping it deferred.
///
/// How adapters chain. Each declares the `dtype` and `shape` its transform produces (the
/// writer needs the resulting byte length up front) while the transform itself runs only once
/// the bytes are finally drawn. `name` is passed explicitly because some adapters rename as
/// they go.
#[cfg(target_has_atomic = "ptr")]
pub fn map_data(
    tensor: &PackTensor,
    name: String,
    dtype: DType,
    shape: Shape,
    f: impl Fn(TensorData) -> TensorData + Send + Sync + 'static,
) -> PackTensor {
    let source = tensor.clone();
    deferred(name, dtype, shape, tensor.param_id, move || {
        to_data(&source).map(&f)
    })
}

/// Wrap a tensor's data in a transform, keeping it deferred.
///
/// See the `target_has_atomic = "ptr"` variant.
#[cfg(not(target_has_atomic = "ptr"))]
pub fn map_data(
    tensor: &PackTensor,
    name: String,
    dtype: DType,
    shape: Shape,
    f: impl Fn(TensorData) -> TensorData + 'static,
) -> PackTensor {
    let source = tensor.clone();
    deferred(name, dtype, shape, tensor.param_id, move || {
        to_data(&source).map(&f)
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use burn_core::tensor::{Bool, Device, Distribution, Int, shape};

    /// `byte_len` is computed from the declared shape and dtype rather than observed, and the
    /// burnpack writer reserves exactly that many bytes before ever running the provider. A
    /// disagreement would silently misplace everything written after this tensor.
    ///
    /// The length is also checked inside the materialization path, so a mismatch surfaces as
    /// an error rather than a wrong number; report both the same way.
    fn assert_byte_len_matches(tensor: PackTensor) {
        let (declared, name) = (tensor.byte_len(), tensor.name.clone());

        match to_data(&tensor) {
            Ok(data) => assert_eq!(
                declared,
                data.bytes.len(),
                "byte_len disagrees with the materialized bytes for {name}"
            ),
            Err(e) => panic!("byte_len disagrees with the materialized bytes for {name}: {e}"),
        }
    }

    #[test]
    fn data_len_covers_every_dtype_family() {
        let device = Device::default();

        let floats = Tensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let ints = Tensor::<2, Int>::from_data([[1, 2], [3, 4]], &device);
        let bools = Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device);

        assert_byte_len_matches(from_tensor(&floats, "float".to_string(), None));
        assert_byte_len_matches(from_tensor(&ints, "int".to_string(), None));
        assert_byte_len_matches(from_tensor(&bools, "bool".to_string(), None));
    }

    /// Quantized is the one family where [`data_len`] reconstructs the layout instead of
    /// deriving it from an element count, so it is where the two drift apart.
    ///
    /// The shapes and schemes here hit the two ways that reconstruction has been wrong: a
    /// value-byte count that is not a multiple of the 4-byte scale alignment (which a spurious
    /// round-up inflates), and a sub-byte `QuantValue` under `QuantStore::Native` (whose stored
    /// width rounds down to zero bytes if divided rather than div_ceil'd). A 32x32 Q8 tensor
    /// passes either way, so it cannot stand in for these.
    ///
    /// Both axes rely on the test backend storing quantized values natively, one `i8` per
    /// value: `quantize_dynamic` rewrites the scheme's store to `QuantStore::Native` even
    /// though `QuantScheme::default()` starts as `PackedU32`. A backend honoring `PackedU32`
    /// would produce 4-byte-exact value counts and exercise neither axis.
    #[test]
    fn data_len_matches_materialized_bytes_when_quantized() {
        let device = Device::default();

        let schemes = [
            ("q8", QuantScheme::default()),
            ("q4", QuantScheme::default().with_value(QuantValue::Q4S)),
            ("q2", QuantScheme::default().with_value(QuantValue::Q2S)),
        ];
        // Element counts that are and are not multiples of the 4-byte scale alignment.
        let shapes = [shape![32, 32], shape![3, 3], shape![5, 5], shape![2, 3]];

        for (name, scheme) in schemes {
            for shape in &shapes {
                let tensor = Tensor::<2>::random(shape.clone(), Distribution::Default, &device)
                    .quantize_dynamic(&scheme);
                assert_byte_len_matches(from_tensor(&tensor, format!("{name}-{shape:?}"), None));
            }
        }
    }

    /// The same agreement against what `TensorData::quantized` writes, over block sizes rather
    /// than through a backend, so the per-block and per-tensor scale arithmetic is pinned
    /// independently of what any backend happens to produce.
    #[test]
    fn data_len_matches_quantized_bytes() {
        let base = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);

        // 8 values in blocks of 4 lands on `QPARAM_ALIGN`, which would hide a padding
        // assumption; 6 in blocks of 3 and 10 in blocks of 5 do not.
        for (values, block) in [(8usize, 4usize), (6, 3), (10, 5)] {
            let scales = vec![0.5f32; values / block];
            let one_level = base.per_block([block as u8], ScaleDtype::UE4M3);
            let two_level = base
                .per_block([block as u8], ScaleDtype::UE4M3)
                .per_tensor(ScaleDtype::F32);

            for (scheme, global) in [(one_level, None), (two_level, Some(3.0f32))] {
                let data =
                    TensorData::quantized(vec![0i8; values], [values], scheme, &scales, global);

                assert_eq!(
                    data_len(data.dtype, &data.shape),
                    data.bytes.len(),
                    "predicted size disagrees with the written bytes for {values} values \
                     in blocks of {block}, {scheme:?}"
                );
            }
        }
    }

    /// Packed quantized storage divides only the packed dimension, so a non-divisible extent
    /// pads once per line rather than once over the flattened tensor. No current backend can
    /// materialize such a tensor, so this pins the formula against the storage shape the
    /// allocation would use (`CubeTensor::quantized_storage`) rather than against real bytes.
    #[test]
    fn data_len_packs_per_line_for_packed_stores() {
        // Q4 in u32 words: 8 values per storage element, packed along the last dimension.
        let scheme = QuantScheme::default().with_value(QuantValue::Q4S);
        let packed = |shape: Shape| data_len(DType::QFloat(scheme), &shape);

        // 3 lines x ceil(3 / 8) = 3 u32 words = 12 value bytes, plus one tensor-level f32
        // scale. Flattening first would say ceil(9 / 8) * 4 = 8.
        assert_eq!(packed(shape![3, 3]), 12 + 4);

        // The same over more than two dimensions: 2 * 5 lines x ceil(9 / 8) = 20 words.
        assert_eq!(packed(shape![2, 5, 9]), 20 * 4 + 4);

        // A divisible extent is unaffected, and agrees with the flattened count.
        assert_eq!(packed(shape![4, 8]), 4 * 4 + 4);
    }

    /// A backend can panic reading a tensor back from its device. That panic must not unwind
    /// through the burnpack writer partway through a container, so every provider built here
    /// is wrapped: the panic becomes an ordinary error the writer can report and abort on.
    #[test]
    fn a_panicking_provider_becomes_an_error() {
        let tensor = deferred("weight".to_string(), DType::F32, shape![2, 2], None, || {
            panic!("device readback panicked")
        });

        let err = to_data(&tensor).expect_err("a panicking provider must not unwind");
        assert!(
            matches!(&err, PackError::ValidationError(m) if m.contains("panic")),
            "expected a validation error naming the panic, got {err:?}"
        );
    }

    /// A provider's own error is passed through untouched; only panics are translated.
    #[test]
    fn a_provider_error_passes_through() {
        let tensor = deferred("weight".to_string(), DType::F32, shape![2, 2], None, || {
            Err(PackError::IoError("simulated IO error".to_string()))
        });

        let err = to_data(&tensor).unwrap_err();
        assert!(
            matches!(&err, PackError::IoError(m) if m == "simulated IO error"),
            "expected the provider's own error, got {err:?}"
        );
    }

    /// Snapshotting a module must not read anything back from the device. This is what bounds
    /// a save's peak host memory by the largest single tensor rather than by the whole model,
    /// and it is invisible in the metadata, so pin it on the call count.
    #[test]
    fn from_tensor_reads_nothing_until_asked() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        // Tests only build for std targets, so the atomic pointer is always available here.
        use alloc::sync::Arc;

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);

        let tensor = deferred(
            "weight".to_string(),
            DType::F32,
            shape![4],
            None,
            move || {
                counter.fetch_add(1, Ordering::Relaxed);
                Ok(data.clone())
            },
        );

        // Metadata alone costs nothing.
        assert_eq!(tensor.byte_len(), 16);
        assert_eq!(tensor.shape, shape![4]);
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        to_data(&tensor).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    /// One generic constructor has to cover all three parameter kinds, since `Basic` is what
    /// supplies `dtype`, `shape` and `to_data` for each.
    #[test]
    fn from_tensor_covers_every_kind() {
        let device = Device::default();

        let floats = from_tensor(
            &Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device),
            "float".to_string(),
            Some(7),
        );
        assert_eq!(floats.name, "float");
        assert_eq!(floats.shape, shape![2, 2]);
        assert_eq!(floats.param_id, Some(7));
        assert_eq!(to_data(&floats).unwrap().shape, shape![2, 2]);

        let ints = from_tensor(
            &Tensor::<2, Int>::from_data([[1, 2], [3, 4]], &device),
            "int".to_string(),
            None,
        );
        assert_eq!(ints.dtype, device.settings().int_dtype.into());

        let bools = from_tensor(
            &Tensor::<2, Bool>::from_data([[true, false], [false, true]], &device),
            "bool".to_string(),
            None,
        );
        assert_eq!(to_data(&bools).unwrap().shape, shape![2, 2]);
    }

    /// `map_data` has to declare the length its transform will produce, not the one it
    /// consumed. A cast that shrinks the data while still claiming the source's length would
    /// be caught only once the writer had already committed its offsets.
    #[test]
    fn map_data_declares_the_transformed_length() {
        let device = Device::default();
        let source = from_tensor(
            &Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device),
            "weight".to_string(),
            None,
        );
        assert_eq!(source.byte_len(), 16);

        let cast = map_data(
            &source,
            "weight".to_string(),
            DType::F16,
            source.shape.clone(),
            |data| data.convert_dtype(DType::F16),
        );

        assert_eq!(cast.byte_len(), 8);
        assert_byte_len_matches(cast);
    }
}
