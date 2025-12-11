//! External data support for ONNX tensors
//!
//! ONNX models larger than 2GB use external data storage due to protobuf's size limit.
//! Tensor data is stored in separate files, referenced via `external_data` fields in TensorProto.
//!
//! See: <https://onnx.ai/onnx/repo-docs/ExternalData.html>

use std::path::{Path, PathBuf};

/// Information about externally stored tensor data
///
/// Parsed from TensorProto's `external_data` field which contains key-value pairs:
/// - `location` (required): POSIX path relative to ONNX file directory
/// - `offset` (optional): Byte position where data begins (default: 0)
/// - `length` (optional): Number of bytes to read (default: entire file from offset)
/// - `checksum` (optional): SHA1 digest for verification (not currently used)
#[derive(Debug, Clone)]
pub struct ExternalDataInfo {
    /// Relative path to external data file (from ONNX file directory)
    pub location: PathBuf,
    /// Byte offset within the file (default: 0)
    pub offset: u64,
    /// Number of bytes to read (None means read to end of file)
    pub length: Option<u64>,
    /// Optional SHA1 checksum (not currently validated)
    #[allow(dead_code)]
    pub checksum: Option<String>,
}

impl ExternalDataInfo {
    /// Parse external data info from ONNX StringStringEntryProto key-value pairs
    ///
    /// # Arguments
    /// * `entries` - Iterator of (key, value) string pairs from TensorProto.external_data
    ///
    /// # Returns
    /// * `Ok(ExternalDataInfo)` if location is present
    /// * `Err(String)` if location is missing
    pub fn from_proto_entries<'a>(
        entries: impl Iterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, String> {
        let mut location: Option<PathBuf> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;
        let mut checksum: Option<String> = None;

        for (key, value) in entries {
            match key {
                "location" => {
                    location = Some(PathBuf::from(value));
                }
                "offset" => {
                    offset = value
                        .parse()
                        .map_err(|e| format!("Invalid offset '{}': {}", value, e))?;
                }
                "length" => {
                    length = Some(
                        value
                            .parse()
                            .map_err(|e| format!("Invalid length '{}': {}", value, e))?,
                    );
                }
                "checksum" => {
                    checksum = Some(value.to_string());
                }
                _ => {
                    // Ignore unknown keys per ONNX spec
                    log::debug!("Ignoring unknown external_data key: {}", key);
                }
            }
        }

        let location = location.ok_or("Missing required 'location' in external_data")?;

        Ok(ExternalDataInfo {
            location,
            offset,
            length,
            checksum,
        })
    }

    /// Resolve the external data path relative to a base directory
    ///
    /// # Arguments
    /// * `base_path` - Directory containing the ONNX file
    ///
    /// # Returns
    /// Absolute path to the external data file
    pub fn resolve_path(&self, base_path: &Path) -> PathBuf {
        base_path.join(&self.location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let entries = vec![("location", "weights.bin")];
        let info = ExternalDataInfo::from_proto_entries(entries.into_iter()).unwrap();

        assert_eq!(info.location, PathBuf::from("weights.bin"));
        assert_eq!(info.offset, 0);
        assert_eq!(info.length, None);
        assert_eq!(info.checksum, None);
    }

    #[test]
    fn test_parse_full() {
        let entries = vec![
            ("location", "model_weights.bin"),
            ("offset", "4096"),
            ("length", "1048576"),
            ("checksum", "abc123"),
        ];
        let info = ExternalDataInfo::from_proto_entries(entries.into_iter()).unwrap();

        assert_eq!(info.location, PathBuf::from("model_weights.bin"));
        assert_eq!(info.offset, 4096);
        assert_eq!(info.length, Some(1048576));
        assert_eq!(info.checksum, Some("abc123".to_string()));
    }

    #[test]
    fn test_missing_location() {
        let entries = vec![("offset", "0")];
        let result = ExternalDataInfo::from_proto_entries(entries.into_iter());

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("location"));
    }

    #[test]
    fn test_invalid_offset() {
        let entries = vec![("location", "weights.bin"), ("offset", "not_a_number")];
        let result = ExternalDataInfo::from_proto_entries(entries.into_iter());

        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_path() {
        let info = ExternalDataInfo {
            location: PathBuf::from("weights/layer1.bin"),
            offset: 0,
            length: None,
            checksum: None,
        };

        let resolved = info.resolve_path(Path::new("/models/bert"));
        assert_eq!(resolved, PathBuf::from("/models/bert/weights/layer1.bin"));
    }
}
