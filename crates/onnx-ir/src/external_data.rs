//! External data support for ONNX tensors
//!
//! ONNX models larger than 2GB use external data storage due to protobuf's size limit.
//! Tensor data is stored in separate files, referenced via `external_data` fields in TensorProto.
//!
//! # Security
//!
//! External data paths come from untrusted ONNX files and could be malicious.
//! We implement path traversal protection to prevent attacks like:
//! - `../../../etc/passwd` (parent directory traversal)
//! - `/etc/passwd` (absolute paths)
//! - Null byte injection (path truncation)
//! - Windows-specific: backslash variants (`..\\`)
//!
//! We use a custom implementation instead of crates like `safe-path` or `safe_path` because:
//! - `safe-path` (crates.io) is Linux-only, designed for container runtimes
//! - `safe_path` (smoelius) is v0.1 with low adoption
//! - Our use case is simple: validate path components before a single file read
//! - We don't need TOCTTOU protection (no concurrent filesystem modifications expected)
//!
//! References:
//! - <https://owasp.org/www-community/attacks/Path_Traversal>
//! - <https://portswigger.net/web-security/file-path-traversal>
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
    /// * `Ok(PathBuf)` - Absolute path to the external data file
    /// * `Err(String)` - If path traversal attack detected
    ///
    /// # Security
    ///
    /// This method validates that the resolved path stays within the base directory
    /// to prevent path traversal attacks. Protected attack vectors:
    ///
    /// - **Absolute paths**: `/etc/passwd` - blocked by `is_absolute()` check
    /// - **Parent traversal**: `../../../etc/passwd` - blocked by `Component::ParentDir` check
    /// - **Windows backslash**: `..\\..\\` - handled by Rust's `Path::components()` which
    ///   normalizes separators on Windows
    /// - **Null bytes**: `weights.bin\0.txt` - blocked by explicit null byte check
    /// - **Symlink escapes**: Caught by `canonicalize()` + `starts_with()` check when
    ///   the paths exist on disk
    pub fn resolve_path(&self, base_path: &Path) -> Result<PathBuf, String> {
        // Check for null bytes which could truncate the path in some contexts
        // (e.g., C APIs, older PHP). Rust's Path handles this safely, but we
        // reject such paths as they indicate malicious intent.
        let path_str = self.location.to_string_lossy();
        if path_str.contains('\0') {
            return Err(format!(
                "Security error: null bytes not allowed in external_data location: {:?}",
                self.location
            ));
        }

        // Reject absolute paths (e.g., /etc/passwd, C:\Windows\System32)
        if self.location.is_absolute() {
            return Err(format!(
                "Security error: absolute paths not allowed in external_data location: {:?}",
                self.location
            ));
        }

        // Reject paths with parent directory references (../ or ..\)
        // Rust's Component::ParentDir handles both Unix and Windows separators
        for component in self.location.components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(format!(
                    "Security error: parent directory references ('..') not allowed in external_data location: {:?}",
                    self.location
                ));
            }
        }

        let resolved = base_path.join(&self.location);

        // Extra safety: canonicalize and verify the resolved path is under base_path
        // This catches edge cases like symlinks pointing outside the base directory
        if let (Ok(canonical_base), Ok(canonical_resolved)) =
            (base_path.canonicalize(), resolved.canonicalize())
            && !canonical_resolved.starts_with(&canonical_base)
        {
            return Err(format!(
                "Security error: external_data path escapes base directory: {:?}",
                self.location
            ));
        }

        Ok(resolved)
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

        let resolved = info.resolve_path(Path::new("/models/bert")).unwrap();
        assert_eq!(resolved, PathBuf::from("/models/bert/weights/layer1.bin"));
    }

    #[test]
    fn test_reject_absolute_path() {
        let info = ExternalDataInfo {
            location: PathBuf::from("/etc/passwd"),
            offset: 0,
            length: None,
            checksum: None,
        };

        let result = info.resolve_path(Path::new("/models/bert"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("absolute paths not allowed"));
    }

    #[test]
    fn test_reject_parent_traversal() {
        let info = ExternalDataInfo {
            location: PathBuf::from("../../../etc/passwd"),
            offset: 0,
            length: None,
            checksum: None,
        };

        let result = info.resolve_path(Path::new("/models/bert"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("parent directory references"));
    }

    #[test]
    fn test_reject_hidden_traversal() {
        // Test path that looks innocent but has .. in the middle
        let info = ExternalDataInfo {
            location: PathBuf::from("weights/../../../etc/passwd"),
            offset: 0,
            length: None,
            checksum: None,
        };

        let result = info.resolve_path(Path::new("/models/bert"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("parent directory references"));
    }

    #[test]
    #[cfg(unix)]
    fn test_reject_null_bytes() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        // Create a path with an embedded null byte (only possible via OsStr on Unix)
        let path_with_null = OsStr::from_bytes(b"weights.bin\x00.txt");
        let info = ExternalDataInfo {
            location: PathBuf::from(path_with_null),
            offset: 0,
            length: None,
            checksum: None,
        };

        let result = info.resolve_path(Path::new("/models/bert"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("null bytes not allowed"));
    }

    #[test]
    #[cfg(windows)]
    fn test_reject_windows_absolute_path() {
        let info = ExternalDataInfo {
            location: PathBuf::from("C:\\Windows\\System32\\config\\SAM"),
            offset: 0,
            length: None,
            checksum: None,
        };

        let result = info.resolve_path(Path::new("C:\\models\\bert"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("absolute paths not allowed"));
    }
}
