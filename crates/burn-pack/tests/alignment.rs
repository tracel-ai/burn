//! Alignment guarantees and buffer-sizing (`size`/`write_into`).

mod common;

use burn_pack::{HEADER_SIZE, Reader, TENSOR_ALIGNMENT, Writer, aligned_data_section_start};
use common::{f32_tensor, read_f32};

#[test]
fn data_section_start_is_aligned() {
    let align = TENSOR_ALIGNMENT as usize;
    for metadata_size in [0usize, 1, 10, 245, 246, 247, 4096] {
        let start = aligned_data_section_start(metadata_size);
        assert_eq!(start % align, 0, "data section start must be 256-aligned");
        assert!(
            start >= HEADER_SIZE + metadata_size,
            "must clear header+metadata"
        );
        assert!(
            start < HEADER_SIZE + metadata_size + align,
            "minimal padding"
        );
    }
}

// Per-tensor offset alignment is verified by an in-crate unit test (it needs the internal
// tensor descriptors); see `src/reader.rs`.

#[test]
fn size_matches_to_bytes_length() {
    let writer = Writer::new(vec![
        f32_tensor("a", &[1.0, 2.0, 3.0], &[3], None),
        f32_tensor("b", &[4.0], &[1], None),
    ]);

    let size = writer.size().unwrap();
    let bytes = writer.into_bytes().unwrap();
    assert_eq!(size, bytes.len());
}

#[test]
fn write_into_matches_to_bytes_and_round_trips() {
    // `write_into` and `into_bytes` each consume the writer, so build one per call.
    let make_writer = || Writer::new(vec![f32_tensor("w", &[1.0, 2.0, 3.0, 4.0], &[2, 2], None)]);

    let mut buffer = vec![0u8; make_writer().size().unwrap()];
    make_writer().write_into(&mut buffer).unwrap();

    let from_to_bytes = make_writer().into_bytes().unwrap();
    assert_eq!(&buffer[..], &from_to_bytes[..]);

    let reader = Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(buffer)).unwrap();
    let tensors = reader.into_tensors().unwrap();
    assert_eq!(read_f32(&tensors[0]), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn write_into_rejects_too_small_buffer() {
    let writer = Writer::new(vec![f32_tensor("w", &[1.0, 2.0], &[2], None)]);
    let mut buffer = vec![0u8; writer.size().unwrap() - 1];
    assert!(writer.write_into(&mut buffer).is_err());
}
