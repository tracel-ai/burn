use super::base::{
    BurnpackError, BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
    TENSOR_ALIGNMENT, TensorDescriptor,
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

/// Align an offset to the specified alignment boundary.
///
/// Returns the smallest value >= `offset` that is a multiple of `alignment`.
#[inline]
const fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

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
        // Build tensor descriptors and calculate offsets with alignment
        let mut tensors = BTreeMap::new();
        let mut current_offset = 0u64;

        for snapshot in &self.snapshots {
            let data_len = snapshot.data_len() as u64;

            // Align the start offset for mmap zero-copy support
            let aligned_start = align_offset(current_offset, TENSOR_ALIGNMENT);
            let end = aligned_start.checked_add(data_len).ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Tensor offset overflow: {} + {} exceeds maximum",
                    aligned_start, data_len
                ))
            })?;

            tensors.insert(
                snapshot.full_path(),
                TensorDescriptor {
                    dtype: snapshot.dtype,
                    shape: snapshot.shape.iter().map(|&s| s as u64).collect(),
                    data_offsets: (aligned_start, end),
                    param_id: snapshot.tensor_id.map(|id| id.val()),
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
    /// The size includes padding bytes for tensor alignment.
    pub fn size(&self) -> Result<usize, BurnpackError> {
        let (metadata, metadata_bytes) = self.build_metadata()?;

        // Calculate total data section size from aligned offsets
        // The last tensor's end offset gives us the total data section size
        let data_size = metadata
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .unwrap_or(0) as usize;

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
        let (metadata, metadata_bytes) = self.build_metadata()?;

        // Check metadata size fits in u32
        let metadata_size: u32 = metadata_bytes.len().try_into().map_err(|_| {
            BurnpackError::IoError(format!(
                "Metadata size {} exceeds maximum of {} bytes",
                metadata_bytes.len(),
                u32::MAX
            ))
        })?;

        // Create header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size,
        };

        // Calculate required size from aligned offsets
        let data_size = metadata
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .unwrap_or(0) as usize;
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

        // Data section start - this is the base for tensor offsets
        let data_section_start = offset;

        // Write tensor data with alignment padding
        for snapshot in &self.snapshots {
            // Get the aligned offset from metadata
            let descriptor = metadata.tensors.get(&snapshot.full_path()).ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Internal error: tensor '{}' not found in metadata",
                    snapshot.full_path()
                ))
            })?;
            let aligned_offset = descriptor.data_offsets.0 as usize;
            let target_offset = data_section_start + aligned_offset;

            // Write padding zeros if needed
            if target_offset > offset {
                buffer[offset..target_offset].fill(0);
                offset = target_offset;
            }

            let expected_len = snapshot.data_len();
            let data = snapshot.to_data().map_err(|e| {
                BurnpackError::IoError(format!("Failed to get tensor data: {:?}", e))
            })?;
            let actual_len = data.bytes.len();

            // Validate data length consistency
            if actual_len != expected_len {
                return Err(BurnpackError::IoError(format!(
                    "Data corruption: tensor '{}' has inconsistent length (expected {}, got {})",
                    snapshot.full_path(),
                    expected_len,
                    actual_len
                )));
            }

            buffer[offset..offset + actual_len].copy_from_slice(&data.bytes);
            offset += actual_len;
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

        let (metadata, metadata_bytes) = self.build_metadata()?;

        // Check metadata size fits in u32
        let metadata_size: u32 = metadata_bytes.len().try_into().map_err(|_| {
            BurnpackError::IoError(format!(
                "Metadata size {} exceeds maximum of {} bytes",
                metadata_bytes.len(),
                u32::MAX
            ))
        })?;

        // Create and write header
        let header = BurnpackHeader {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size,
        };

        file.write_all(&header.into_bytes())
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Write metadata
        file.write_all(&metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Track current position in data section
        let mut data_offset = 0usize;

        // Stream tensor data directly to file with alignment padding
        for snapshot in &self.snapshots {
            // Get the aligned offset from metadata
            let descriptor = metadata.tensors.get(&snapshot.full_path()).ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Internal error: tensor '{}' not found in metadata",
                    snapshot.full_path()
                ))
            })?;
            let aligned_offset = descriptor.data_offsets.0 as usize;

            // Write padding zeros if needed
            if aligned_offset > data_offset {
                let padding_size = aligned_offset - data_offset;
                let padding = vec![0u8; padding_size];
                file.write_all(&padding)
                    .map_err(|e| BurnpackError::IoError(e.to_string()))?;
                data_offset = aligned_offset;
            }

            let expected_len = snapshot.data_len();
            let data = snapshot.to_data().map_err(|e| {
                BurnpackError::IoError(format!("Failed to get tensor data: {:?}", e))
            })?;
            let actual_len = data.bytes.len();

            // Validate data length consistency
            if actual_len != expected_len {
                return Err(BurnpackError::IoError(format!(
                    "Data corruption: tensor '{}' has inconsistent length (expected {}, got {})",
                    snapshot.full_path(),
                    expected_len,
                    actual_len
                )));
            }

            file.write_all(&data.bytes)
                .map_err(|e| BurnpackError::IoError(e.to_string()))?;
            data_offset += actual_len;
        }

        file.flush()
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        Ok(())
    }
}
