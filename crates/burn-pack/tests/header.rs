//! Header encoding / decoding and format constants.

use burn_pack::{Error, FORMAT_VERSION, HEADER_SIZE, Header, MAGIC_NUMBER};

#[test]
fn header_constants() {
    assert_eq!(HEADER_SIZE, 10);
    // "BURN" in ASCII.
    assert_eq!(MAGIC_NUMBER, 0x4255524E);
    assert_eq!(MAGIC_NUMBER.to_le_bytes(), *b"NRUB");
}

#[test]
fn header_round_trip() {
    let header = Header::new(1234);
    let bytes = header.into_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE);

    let decoded = Header::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.magic, MAGIC_NUMBER);
    assert_eq!(decoded.version, FORMAT_VERSION);
    assert_eq!(decoded.metadata_size, 1234);
}

#[test]
fn header_rejects_bad_magic() {
    let bad = Header {
        magic: 0xDEAD_BEEF,
        version: FORMAT_VERSION,
        metadata_size: 0,
    };
    assert!(matches!(
        Header::from_bytes(&bad.into_bytes()),
        Err(Error::InvalidMagicNumber)
    ));
}

#[test]
fn header_rejects_short_input() {
    assert!(matches!(
        Header::from_bytes(&[0u8; 4]),
        Err(Error::InvalidHeader)
    ));
}
