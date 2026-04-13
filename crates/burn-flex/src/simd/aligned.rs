//! SIMD-aligned memory allocation utilities.
//!
//! Provides aligned vectors for optimal SIMD performance. Uses 64-byte alignment
//! to support all SIMD instruction sets (NEON: 16B, AVX2: 32B, AVX-512: 64B).

use aligned_vec::{AVec, ConstAlign};
use alloc::vec::Vec;

/// Alignment for SIMD operations (64 bytes = AVX-512 width).
/// Also works for NEON (16B) and AVX2 (32B) since 64 is a multiple of both.
pub const SIMD_ALIGN: usize = 64;

/// Type alias for 64-byte aligned vector.
pub type AlignedVec<T> = AVec<T, ConstAlign<SIMD_ALIGN>>;

/// Allocate a SIMD-aligned vector with the given capacity.
#[inline]
pub fn alloc_aligned<T>(capacity: usize) -> AlignedVec<T> {
    AVec::with_capacity(SIMD_ALIGN, capacity)
}

/// Allocate a SIMD-aligned vector filled with zeros.
#[inline]
pub fn alloc_aligned_zeroed<T: bytemuck::Zeroable + Clone>(len: usize) -> AlignedVec<T> {
    let mut vec = AVec::with_capacity(SIMD_ALIGN, len);
    vec.resize(len, T::zeroed());
    vec
}

/// Convert an aligned vector to a regular Vec.
///
/// This may copy the data if the alignment requirements differ.
#[inline]
pub fn to_vec<T: Clone>(aligned: AlignedVec<T>) -> Vec<T> {
    aligned.to_vec()
}

/// Convert a slice to an aligned vector (copies data).
#[inline]
pub fn from_slice<T: Clone>(slice: &[T]) -> AlignedVec<T> {
    let mut vec = AVec::with_capacity(SIMD_ALIGN, slice.len());
    vec.extend_from_slice(slice);
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let vec: AlignedVec<f32> = alloc_aligned(1000);
        assert_eq!(vec.as_ptr() as usize % SIMD_ALIGN, 0);
    }

    #[test]
    fn test_zeroed() {
        let vec: AlignedVec<f32> = alloc_aligned_zeroed(100);
        assert_eq!(vec.len(), 100);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let aligned = from_slice(&data);
        assert_eq!(aligned.as_ptr() as usize % SIMD_ALIGN, 0);
        assert_eq!(&aligned[..], &data[..]);
    }
}
