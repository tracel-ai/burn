use super::base::{
    BurnpackError, BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
    TensorDescriptor,
};
use crate::TensorSnapshot;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Write;
#[cfg(feature = "std")]
use std::path::Path;

/// Writer for creating Burnpack files
pub struct BurnpackWriter {
    /// Tensors to write
    pub(crate) snapshots: Vec<TensorSnapshot>,
    /// Metadata key-value pairs
    pub(crate) metadata: BTreeMap<String, String>,
}

impl BurnpackWriter {
    /// Create a new writer
    pub fn new(snapshots: Vec<TensorSnapshot>) -> Self {
        Self {
            snapshots,
            metadata: BTreeMap::new(),
        }
    }

    /// Builder pattern: add metadata and return self
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Write to a byte buffer
    pub fn to_bytes(&self) -> Result<Vec<u8>, BurnpackError> {
        // Build tensor descriptors and calculate offsets
        let mut tensors = BTreeMap::new();
        let mut current_offset = 0u64;

        for snapshot in &self.snapshots {
            let data_len = snapshot.data_len() as u64;
            let start = current_offset;
            let end = start + data_len;

            tensors.insert(
                snapshot.full_path(),
                TensorDescriptor {
                    dtype: snapshot.dtype,
                    shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
                    data_offsets: (start, end),
                },
            );

            current_offset = end;
        }

        // Create metadata structure
        let metadata = BurnpackMetadata {
            tensors,
            metadata: self.metadata.clone(),
        };

        // Serialize metadata with CBOR
        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Create header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size: metadata_bytes.len() as u32,
        };

        // Calculate total size
        let data_size = self.snapshots.iter().map(|s| s.data_len()).sum::<usize>();
        let total_size = HEADER_SIZE + metadata_bytes.len() + data_size;

        // Build the final buffer
        let mut buffer = Vec::with_capacity(total_size);

        // Write header
        buffer.extend_from_slice(&header.to_bytes());

        // Write metadata
        buffer.extend_from_slice(&metadata_bytes);

        // Write tensor data
        for snapshot in &self.snapshots {
            let data = snapshot.to_data().map_err(|e| {
                BurnpackError::IoError(format!("Failed to get tensor data: {:?}", e))
            })?;
            buffer.extend_from_slice(&data.bytes);
        }

        Ok(buffer)
    }

    /// Write directly to a file (more memory efficient for large models)
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BurnpackError> {
        let mut file = File::create(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Build tensor descriptors and calculate offsets
        let mut tensors = BTreeMap::new();
        let mut current_offset = 0u64;

        for snapshot in &self.snapshots {
            let data_len = snapshot.data_len() as u64;
            let start = current_offset;
            let end = start + data_len;

            tensors.insert(
                snapshot.full_path(),
                TensorDescriptor {
                    dtype: snapshot.dtype,
                    shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
                    data_offsets: (start, end),
                },
            );

            current_offset = end;
        }

        // Create metadata
        let metadata = BurnpackMetadata {
            tensors,
            metadata: self.metadata.clone(),
        };

        // Serialize metadata
        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Create and write header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size: metadata_bytes.len() as u32,
        };

        file.write_all(&header.to_bytes())
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Write metadata
        file.write_all(&metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Stream tensor data directly to file
        for snapshot in &self.snapshots {
            let data = snapshot.to_data().map_err(|e| {
                BurnpackError::IoError(format!("Failed to get tensor data: {:?}", e))
            })?;
            file.write_all(&data.bytes)
                .map_err(|e| BurnpackError::IoError(e.to_string()))?;
        }

        file.flush()
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        Ok(())
    }
}
