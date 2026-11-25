//! The read a tuning round runs before a sample, held to reading all of its scratch.
//!
//! A memory-bound candidate is measured after the autotune reads past the last-level cache
//! ([`touch`](super::touch)), so a weight the cache would otherwise serve is timed
//! the way a decode step meets it. A read that covers only part of the scratch leaves the cache
//! warm and the round elects on cache rates. Coverage is checked as a sum, not a stopwatch: a
//! device's caches cannot flatter a wrong sum.

use super::read_through;
use crate::CubeDevice;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::{Shape, metadata::Metadata};

/// Vectors each unit sums into the one it writes, as [`read_through`]'s doc states it.
const SUMMED: usize = 64;

/// A value each element carries that no partial or repeated read sums to: distinct along the
/// scratch, exact in `f32` at every partial sum below.
fn value(i: usize) -> f32 {
    ((i * 7919) % 4093) as f32
}

#[test]
fn the_cold_read_reads_every_vector_of_its_scratch_once() {
    let client = CubeDevice::default().client();
    let elem = ElemType::Float(FloatKind::F32);
    // Not a multiple of the sum's reach, so the last units read a short run.
    let elements = SUMMED * 4096 * 3 + 4 * 37;
    let host: Vec<f32> = (0..elements).map(value).collect();
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scratch = TensorHandle::from_metadata(
        client.create_from_slice(&bytes),
        Metadata::new(Shape::new([elements]), vec![1]),
        elem,
    );

    let output = read_through(&client, scratch);
    let written = output.shape()[0];
    let sums: Vec<f32> = client
        .read_one_unchecked(output.handle)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();

    assert!(written * SUMMED >= elements);
    for (at, sum) in sums[..written].iter().enumerate() {
        let expected: f32 = (0..SUMMED)
            .map(|i| at + i * written)
            .filter(|&index| index < elements)
            .map(value)
            .sum();
        assert_eq!(
            *sum, expected,
            "element {at} of the read sums the wrong vectors: {sum} against {expected}"
        );
    }
}
