use super::base::{
    BurnpackError, BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
    TensorDescriptor,
};
use crate::TensorSnapshot;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use burn_tensor::Bytes;

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

    /// Build tensor descriptors and metadata
    fn build_metadata(&self) -> Result<(BurnpackMetadata, Vec<u8>), BurnpackError> {
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

        Ok((metadata, metadata_bytes))
    }

    /// Calculate the total size needed for the burnpack data
    ///
    /// This is useful when you want to pre-allocate a buffer for `write_into()`.
    pub fn size(&self) -> Result<usize, BurnpackError> {
        let (_, metadata_bytes) = self.build_metadata()?;
        let data_size = self.snapshots.iter().map(|s| s.data_len()).sum::<usize>();
        Ok(HEADER_SIZE + metadata_bytes.len() + data_size)
    }

    /// Write burnpack data into a caller-provided buffer
    ///
    /// The buffer must be large enough to hold all data. Use `size()` to determine
    /// the required buffer size. If the buffer is too small, this will return an error.
    ///
    /// This allows the caller to control buffer allocation, enabling optimizations like:
    /// - Buffer reuse across multiple writes
    /// - Custom allocators
    /// - Pinned memory for GPU transfers
    ///
    /// # Arguments
    ///
    /// * `buffer` - Mutable slice to write data into. Must be at least `size()` bytes.
    pub fn write_into(&self, buffer: &mut [u8]) -> Result<(), BurnpackError> {
        let (_, metadata_bytes) = self.build_metadata()?;

        // Create header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size: metadata_bytes.len() as u32,
        };

        // Calculate required size
        let data_size = self.snapshots.iter().map(|s| s.data_len()).sum::<usize>();
        let total_size = HEADER_SIZE + metadata_bytes.len() + data_size;

        // Check buffer size
        if buffer.len() < total_size {
            return Err(BurnpackError::IoError(format!(
                "Buffer too small: need {} bytes, got {} bytes",
                total_size,
                buffer.len()
            )));
        }

        let mut offset = 0;

        // Write header
        let header_bytes = header.into_bytes();
        buffer[offset..offset + HEADER_SIZE].copy_from_slice(&header_bytes);
        offset += HEADER_SIZE;

        // Write metadata
        buffer[offset..offset + metadata_bytes.len()].copy_from_slice(&metadata_bytes);
        offset += metadata_bytes.len();

        // Write tensor data
        for snapshot in &self.snapshots {
            let data = snapshot.to_data().map_err(|e| {
                BurnpackError::IoError(format!("Failed to get tensor data: {:?}", e))
            })?;
            let data_len = data.bytes.len();
            buffer[offset..offset + data_len].copy_from_slice(&data.bytes);
            offset += data_len;
        }

        Ok(())
    }

    /// Write to a byte buffer (convenience method)
    ///
    /// This allocates a buffer internally and writes the burnpack data.
    /// For more control over buffer allocation, use `size()` + `write_into()`.
    pub fn to_bytes(&self) -> Result<Bytes, BurnpackError> {
        let size = self.size()?;
        let mut buffer = vec![0u8; size];
        self.write_into(&mut buffer)?;
        Ok(Bytes::from_bytes_vec(buffer))
    }

    /// Write directly to a file (more memory efficient for large models)
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BurnpackError> {
        let mut file = File::create(path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let (_, metadata_bytes) = self.build_metadata()?;

        // Create and write header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size: metadata_bytes.len() as u32,
        };

        file.write_all(&header.into_bytes())
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
