use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;

use crate::{DType, SplitTensorData, TensorData};

/// Converts a real float `TensorData` into interleaved complex `TensorData` by inserting zero
/// imaginary parts.
///
/// Each element `r` in `data` becomes `[r, 0]` in the output, and the dtype is promoted from
/// the real float dtype (e.g., `F32`) to the corresponding complex dtype (e.g., `Complex32`).
///
/// # Arguments
///
/// * `data` - A real float `TensorData`.
///
/// # Returns
///
/// A `TensorData` with the same shape and an interleaved complex dtype, where the imaginary
/// parts are all zero.
#[inline]
pub fn interleaved_data_from_real_data(data: TensorData) -> TensorData {
    let elem_size = data.dtype.size();

    let mut interleaved_bytes = zeroed_vec(data.bytes.len() * 2);

    // Create chunks of [Real, Imag] (size is 2 * elem_size)
    // We only iterate over the source 'data.bytes' in elem_size steps
    let src_chunks = data.bytes.chunks_exact(elem_size);
    let dest_chunks = interleaved_bytes.chunks_exact_mut(elem_size * 2);

    for (src, dest) in src_chunks.zip(dest_chunks) {
        // dest[0..elem_size] is Real, dest[elem_size..] is Imag (already 0)
        dest[..elem_size].copy_from_slice(src);
    }
    TensorData::from_bytes_vec(
        interleaved_bytes,
        data.shape.clone(),
        real_to_complex_dtype(data.dtype),
    )
}
#[inline]
fn zeroed_vec(size: usize) -> Vec<u8> {
    let mut vec = Vec::with_capacity(size);
    vec.extend(core::iter::repeat_n(0, size));
    vec
}

/// Converts an imaginary float `TensorData` into interleaved complex `TensorData` by inserting
/// zero real parts.
///
/// Each element `i` in `data` becomes `[0, i]` in the output, and the dtype is promoted from
/// the real float dtype (e.g., `F32`) to the corresponding complex dtype (e.g., `Complex32`).
///
/// # Arguments
///
/// * `data` - A float `TensorData` representing imaginary components.
///
/// # Returns
///
/// A `TensorData` with the same shape and an interleaved complex dtype, where the real
/// parts are all zero.
#[inline]
pub fn interleaved_data_from_imag_data(data: TensorData) -> TensorData {
    let elem_size = data.dtype.size();
    let mut interleaved_bytes = zeroed_vec(data.bytes.len() * 2);

    // Create chunks of [Real, Imag] (size is 2 * elem_size)
    // We only iterate over the source 'data.bytes' in elem_size steps
    let src_chunks = data.bytes.chunks_exact(elem_size);
    let dest_chunks = interleaved_bytes.chunks_exact_mut(elem_size * 2);

    for (src, dest) in src_chunks.zip(dest_chunks) {
        // dest[0..elem_size] is Real (already 0), dest[elem_size..] is Imag
        dest[elem_size..].copy_from_slice(src);
    }
    TensorData::from_bytes_vec(
        interleaved_bytes,
        data.shape.clone(),
        real_to_complex_dtype(data.dtype),
    )
}

/// Combines separate real and imaginary `TensorData` into a single interleaved complex
/// `TensorData`.
///
/// Each element pair `(r, i)` becomes `[r, i]` in the output. Both inputs must have the same
/// shape and dtype. The output dtype is promoted to the corresponding complex dtype.
///
/// # Arguments
///
/// * `real` - A float `TensorData` containing the real components.
/// * `imag` - A float `TensorData` containing the imaginary components.
///
/// # Returns
///
/// A `TensorData` with the same shape and an interleaved complex dtype.
#[inline]
pub fn interleaved_data_from_parts_data(real: TensorData, imag: TensorData) -> TensorData {
    let elem_size = real.dtype.size();
    // Pre-allocate the full size
    let mut interleaved_bytes = zeroed_vec(real.bytes.len() * 2);

    let real_chunks = real.bytes.chunks_exact(elem_size);
    let imag_chunks = imag.bytes.chunks_exact(elem_size);
    let dest_chunks = interleaved_bytes.chunks_exact_mut(elem_size * 2);

    // Zip all three: Real src, Imag src, and the interleaved dest
    for ((r_src, i_src), d_chunk) in real_chunks.zip(imag_chunks).zip(dest_chunks) {
        d_chunk[..elem_size].copy_from_slice(r_src);
        d_chunk[elem_size..].copy_from_slice(i_src);
    }

    TensorData::from_bytes_vec(
        interleaved_bytes,
        real.shape.clone(),
        real_to_complex_dtype(real.dtype),
    )
}

/// Extracts the real components from an interleaved complex `TensorData`.
///
/// Each complex element `[r, i]` in the input yields only `r` in the output. The dtype is
/// demoted from the complex dtype (e.g., `Complex32`) to the corresponding real float dtype
/// (e.g., `F32`).
///
/// # Arguments
///
/// * `interleaved` - A complex `TensorData` in interleaved layout.
///
/// # Returns
///
/// A float `TensorData` with the same shape containing only the real parts.
#[inline]
pub fn interleaved_data_to_real_data(interleaved: TensorData) -> TensorData {
    let real_dtype = complex_to_real_dtype(interleaved.dtype);
    let real_elem_size: usize = real_dtype.size();
    let complex_elem_size = interleaved.dtype.size();

    let mut real_bytes = Vec::with_capacity(interleaved.bytes.len() / 2);

    // Each "element" in the interleaved data contains [Real, Imag]
    for chunk in interleaved.bytes.chunks_exact(complex_elem_size) {
        // Grab only the first half of the complex element's bytes
        let real_part = &chunk[..real_elem_size];
        real_bytes.extend_from_slice(real_part);
    }

    // Shape remains exactly the same; only the DType is updated to the "inner" type
    TensorData::from_bytes_vec(real_bytes, interleaved.shape.clone(), real_dtype)
}

/// Converts an interleaved complex `TensorData` into a float `TensorData`, flattening the real and imaginary parts into a single float buffer.
#[inline]
pub fn interleaved_data_to_raw_float_data(interleaved: TensorData) -> TensorData {
    let real_dtype = complex_to_real_dtype(interleaved.dtype);

    let out_shape = {
        let mut s = interleaved.shape.clone();
        s.push(2); // Add a new innermost dimension of size 2 for [Real, Imag]
        s
    };

    // Shape is updated to include the new innermost dimension of size 2 for [Real, Imag]
    TensorData::from_bytes(interleaved.bytes, out_shape, real_dtype)
}

/// Extracts the imaginary components from an interleaved complex `TensorData`.
///
/// Each complex element `[r, i]` in the input yields only `i` in the output. The dtype is
/// demoted from the complex dtype (e.g., `Complex32`) to the corresponding real float dtype
/// (e.g., `F32`).
///
/// # Arguments
///
/// * `interleaved` - A complex `TensorData` in interleaved layout.
///
/// # Returns
///
/// A float `TensorData` with the same shape containing only the imaginary parts.
#[inline]
pub fn interleaved_data_to_imag_data(interleaved: TensorData) -> TensorData {
    let real_dtype = complex_to_real_dtype(interleaved.dtype);
    let real_elem_size = real_dtype.size();
    let complex_elem_size = interleaved.dtype.size();

    let mut real_bytes = Vec::with_capacity(interleaved.bytes.len() / 2);

    // Each "element" in the interleaved data contains [Real, Imag]
    for chunk in interleaved.bytes.chunks_exact(complex_elem_size) {
        // Grab only the first half of the complex element's bytes
        let real_part = &chunk[real_elem_size..];
        real_bytes.extend_from_slice(real_part);
    }

    // Shape remains exactly the same; only the DType is updated to the "inner" type
    TensorData::from_bytes_vec(real_bytes, interleaved.shape.clone(), real_dtype)
}

/// Splits an interleaved complex `TensorData` into a [`SplitTensorData`] containing separate
/// real and imaginary byte buffers.
///
/// Each complex element `[r, i]` is split so that `r` goes into `real_bytes` and `i` into
/// `imag_bytes`. The shape is preserved and the dtype is demoted to the corresponding real
/// float dtype.
///
/// # Arguments
///
/// * `interleaved` - A complex `TensorData` in interleaved layout.
///
/// # Returns
///
/// A [`SplitTensorData`] containing the real and imaginary byte buffers with the same shape.
#[inline]
pub fn split_from_interleaved_data(interleaved: TensorData) -> SplitTensorData {
    let real_dtype = complex_to_real_dtype(interleaved.dtype);
    let real_elem_size = real_dtype.size();
    let complex_elem_size = interleaved.dtype.size(); // This should be 2 * real_elem_size

    let mut real_bytes = Vec::with_capacity(interleaved.bytes.len() / 2);
    let mut imag_bytes = Vec::with_capacity(interleaved.bytes.len() / 2);

    // Each "element" in the interleaved data contains [Real, Imag]
    for chunk in interleaved.bytes.chunks_exact(complex_elem_size) {
        // Grab only the first half of the complex element's bytes
        real_bytes.extend_from_slice(&chunk[..real_elem_size]);
        imag_bytes.extend_from_slice(&chunk[real_elem_size..]);
    }

    SplitTensorData {
        real_bytes: Bytes::from_bytes_vec(real_bytes),
        imag_bytes: Bytes::from_bytes_vec(imag_bytes),
        shape: interleaved.shape,
        dtype: real_dtype,
    }
}

/// Maps a real float [`DType`] to the corresponding complex [`DType`].
///
/// * `F32` → `Complex32`
/// * `F64` → `Complex64`
///
/// # Panics
///
/// Panics if `real_data` is not a supported float dtype.
#[inline(always)]
pub const fn real_to_complex_dtype(real_data: DType) -> DType {
    match real_data {
        DType::F32 => DType::Complex32,
        DType::F64 => DType::Complex64,
        _ => panic!("r2c: Unsupported dtype"),
    }
}

/// Maps a complex [`DType`] to the corresponding real float [`DType`].
///
/// * `Complex32` → `F32`
/// * `Complex64` → `F64`
///
/// # Panics
///
/// Panics if `real_data` is not a supported complex dtype.
#[inline(always)]
pub const fn complex_to_real_dtype(real_data: DType) -> DType {
    match real_data {
        DType::Complex32 => DType::F32,
        DType::Complex64 => DType::F64,
        _ => panic!("c2r: Unsupported dtype"),
    }
}
