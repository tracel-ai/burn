//! Malformed / malicious input handling.

mod common;

use burn_pack::{
    Bytes, Error, FORMAT_VERSION, Header, MAGIC_NUMBER, MAX_METADATA_SIZE, Reader, Writer,
};
use common::f32_tensor;

fn header_bytes(version: u16, metadata_size: u32) -> Bytes {
    let header = Header {
        magic: MAGIC_NUMBER,
        version,
        metadata_size,
    };
    Bytes::from_bytes_vec(header.into_bytes().to_vec())
}

#[test]
fn rejects_too_short_input() {
    assert!(matches!(
        Reader::from_bytes(Bytes::from_bytes_vec(vec![0u8; 4])),
        Err(Error::InvalidHeader)
    ));
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![0u8; 10];
    bytes[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    assert!(matches!(
        Reader::from_bytes(Bytes::from_bytes_vec(bytes)),
        Err(Error::InvalidMagicNumber)
    ));
}

#[test]
fn rejects_future_version() {
    let bytes = header_bytes(FORMAT_VERSION + 1, 0);
    assert!(matches!(
        Reader::from_bytes(bytes),
        Err(Error::InvalidVersion)
    ));
}

#[test]
fn rejects_oversized_metadata_claim() {
    // The reader bails out on the metadata-size claim before allocating for it.
    let bytes = header_bytes(FORMAT_VERSION, MAX_METADATA_SIZE + 1);
    assert!(matches!(
        Reader::from_bytes(bytes),
        Err(Error::ValidationError(_))
    ));
}

#[test]
fn rejects_metadata_size_past_eof() {
    // Header claims more metadata than the buffer actually contains.
    let bytes = header_bytes(FORMAT_VERSION, 4096);
    assert!(Reader::from_bytes(bytes).is_err());
}

#[test]
fn rejects_duplicate_tensor_names() {
    // Descriptors are keyed by name but data is written from the tensor list: a duplicate
    // name must be rejected up front, not silently corrupt the container.
    let writer = Writer::new(vec![
        f32_tensor("w", &[1.0, 2.0], &[2], None),
        f32_tensor("w", &[3.0, 4.0], &[2], None),
    ]);
    assert!(matches!(
        writer.into_bytes(),
        Err(Error::ValidationError(_))
    ));
}

#[test]
fn rejects_truncated_data_section() {
    // A valid pack, truncated well into its data section, must be rejected (not silently
    // read). Use a large tensor and drop half the file so we are unambiguously below the
    // size the metadata claims.
    let values: Vec<f32> = (0..512).map(|i| i as f32).collect();
    let packed = Writer::new(vec![f32_tensor("w", &values, &[512], None)])
        .into_bytes()
        .unwrap();

    let slice: &[u8] = &packed;
    let mut bytes = slice.to_vec();
    bytes.truncate(bytes.len() / 2);

    assert!(matches!(
        Reader::from_bytes(Bytes::from_bytes_vec(bytes)),
        Err(Error::ValidationError(_))
    ));
}
