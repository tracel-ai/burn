#[cfg(feature = "std")]
use super::base::MAX_FILE_SIZE;
use super::base::{
    BurnpackError, BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
    MAX_CBOR_RECURSION_DEPTH, MAX_METADATA_SIZE, MAX_TENSOR_COUNT, MAX_TENSOR_SIZE,
    aligned_data_section_start,
};
use crate::TensorSnapshot;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::{Bytes, Shape, TensorData};

#[cfg(feature = "std")]
use std::cell::RefCell;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::{Read, Seek};
#[cfg(feature = "std")]
use std::path::Path;

/// Storage backend for BurnpackReader
pub(crate) enum StorageBackend {
    /// Memory-based storage (also used for memory-mapped files converted to bytes::Bytes)
    Memory(Rc<Bytes>),
    /// File-based storage with buffered reading
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    FileBuffered { file: Rc<RefCell<File>> },
}

impl StorageBackend {
    /// Read data from storage into the provided buffer at the given offset.
    ///
    /// # Arguments
    /// * `bytes` - The buffer to read into (caller-allocated)
    /// * `offset` - Absolute file/data position to start reading from
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The requested data range is out of bounds
    /// - Less data is available than requested (indicates corruption or incorrect offset)
    /// - File I/O fails
    ///
    /// # Notes
    ///
    /// The caller allocates the buffer, which allows for buffer reuse and future optimizations
    /// like memory pools and pinned memory.
    ///
    /// This method ensures all backends have consistent behavior: if the exact number of
    /// requested bytes cannot be read, an error is returned to prevent data corruption.
    pub(crate) fn read_into(&self, bytes: &mut [u8], offset: usize) -> Result<(), BurnpackError> {
        match self {
            StorageBackend::Memory(data) => {
                let data_bytes = data.as_ref();
                let end = offset.checked_add(bytes.len()).ok_or_else(|| {
                    BurnpackError::IoError(format!(
                        "Offset overflow: offset {} + length {} exceeds maximum",
                        offset,
                        bytes.len()
                    ))
                })?;

                if end > data_bytes.len() {
                    return Err(BurnpackError::IoError(format!(
                        "Read out of bounds: requested {}..{} but data length is {}",
                        offset,
                        end,
                        data_bytes.len()
                    )));
                }

                bytes.copy_from_slice(&data_bytes[offset..end]);
                Ok(())
            }
            #[cfg(feature = "std")]
            StorageBackend::FileBuffered { file } => {
                use std::io::SeekFrom;

                let mut file = file.borrow_mut();
                file.seek(SeekFrom::Start(offset as u64)).map_err(|e| {
                    BurnpackError::IoError(format!("Failed to seek in file: {}", e))
                })?;

                file.read_exact(bytes).map_err(|e| {
                    BurnpackError::IoError(format!("Failed to read from file: {}", e))
                })?;
                Ok(())
            }
        }
    }

    /// Get full data reference for raw access
    #[allow(dead_code)]
    pub(crate) fn as_bytes(&self) -> Result<&[u8], BurnpackError> {
        match self {
            StorageBackend::Memory(data) => Ok(data.as_ref()),
            #[cfg(feature = "std")]
            StorageBackend::FileBuffered { .. } => Err(BurnpackError::IoError(
                "Cannot get full bytes reference for FileBuffered backend".into(),
            )),
        }
    }

    /// Attempt to slice bytes without copying (zero-copy).
    ///
    /// This uses `Bytes::clone()` + `split()` which is zero-copy when the underlying
    /// `Bytes` was created via `Bytes::from_shared()` (backed by `bytes::Bytes`).
    ///
    /// # Returns
    /// - `Ok(bytes)` - Successfully created a zero-copy slice
    /// - `Err(_)` - Backend doesn't support zero-copy or split failed
    pub(crate) fn slice_bytes(&self, start: usize, end: usize) -> Result<Bytes, BurnpackError> {
        if end < start {
            return Err(BurnpackError::IoError(format!(
                "Invalid slice range: end ({}) < start ({})",
                end, start
            )));
        }

        match self {
            StorageBackend::Memory(data) => {
                // Clone the Bytes - cheap if backed by SharedBytesAllocationController
                let cloned = (**data).clone();

                // Split at start offset to get (_, right)
                let (_, right) = cloned.split(start).map_err(|(_, e)| {
                    BurnpackError::IoError(format!("Failed to split at start {}: {:?}", start, e))
                })?;

                // Split right at (end - start) to get (middle, _)
                let slice_len = end - start;
                let (middle, _) = right.split(slice_len).map_err(|(_, e)| {
                    BurnpackError::IoError(format!(
                        "Failed to split at length {}: {:?}",
                        slice_len, e
                    ))
                })?;

                Ok(middle)
            }
            #[cfg(feature = "std")]
            StorageBackend::FileBuffered { .. } => Err(BurnpackError::IoError(
                "Zero-copy not supported for buffered file reading. Use from_file() with memmap feature for zero-copy loading.".into(),
            )),
        }
    }
}

/// Reader for loading Burnpack files
pub struct BurnpackReader {
    /// Parsed metadata
    pub(crate) metadata: BurnpackMetadata,
    /// Storage backend
    pub(crate) storage: StorageBackend,
    /// Offset to the start of tensor data
    pub(crate) data_offset: usize,
}

impl BurnpackReader {
    /// Load from bytes
    pub fn from_bytes(bytes: Bytes) -> Result<Self, BurnpackError> {
        // Validate minimum size
        if bytes.len() < HEADER_SIZE {
            return Err(BurnpackError::InvalidHeader);
        }

        // Parse header
        let header = BurnpackHeader::from_bytes(&bytes[..HEADER_SIZE])?;

        // Verify magic number
        if header.magic != MAGIC_NUMBER {
            return Err(BurnpackError::InvalidMagicNumber);
        }

        // Verify version compatibility
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Validate metadata size against security limit
        if header.metadata_size > MAX_METADATA_SIZE {
            return Err(BurnpackError::ValidationError(format!(
                "Metadata size {} exceeds maximum allowed size of {} bytes (potential DoS attack)",
                header.metadata_size, MAX_METADATA_SIZE
            )));
        }

        // Parse metadata
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start
            .checked_add(header.metadata_size as usize)
            .ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Metadata size overflow: {} + {}",
                    metadata_start, header.metadata_size
                ))
            })?;

        if bytes.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata = ciborium::de::from_reader_with_recursion_limit(
            &bytes[metadata_start..metadata_end],
            MAX_CBOR_RECURSION_DEPTH,
        )
        .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        // Validate tensor count against security limit
        if metadata.tensors.len() > MAX_TENSOR_COUNT {
            return Err(BurnpackError::ValidationError(format!(
                "File contains {} tensors, exceeding maximum of {} (potential DoS attack)",
                metadata.tensors.len(),
                MAX_TENSOR_COUNT
            )));
        }

        // Validate total file size - ensure file is large enough for all claimed tensor data
        if !metadata.tensors.is_empty() {
            let max_data_offset = metadata
                .tensors
                .values()
                .map(|t| t.data_offsets.1)
                .max()
                .unwrap_or(0);

            let max_data_offset_usize: usize = max_data_offset.try_into().map_err(|_| {
                BurnpackError::ValidationError(format!(
                    "Data offset {} exceeds platform maximum",
                    max_data_offset
                ))
            })?;

            let min_file_size =
                metadata_end
                    .checked_add(max_data_offset_usize)
                    .ok_or_else(|| {
                        BurnpackError::ValidationError("File size calculation overflow".into())
                    })?;

            if bytes.len() < min_file_size {
                return Err(BurnpackError::ValidationError(format!(
                    "File truncated: expected at least {} bytes, got {} bytes",
                    min_file_size,
                    bytes.len()
                )));
            }
        }

        Ok(Self {
            metadata,
            storage: StorageBackend::Memory(Rc::new(bytes)),
            data_offset: aligned_data_section_start(header.metadata_size as usize),
        })
    }

    /// Load from file with memory mapping (most efficient for large files)
    #[cfg(all(feature = "std", feature = "memmap"))]
    pub(crate) fn from_file_mmap<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        let file = File::open(&path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Validate maximum file size to prevent resource exhaustion
        let file_size = file
            .metadata()
            .map_err(|e| BurnpackError::IoError(e.to_string()))?
            .len();

        if file_size > MAX_FILE_SIZE {
            return Err(BurnpackError::ValidationError(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                file_size, MAX_FILE_SIZE
            )));
        }

        // Memory map the file
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| BurnpackError::IoError(e.to_string()))?
        };

        // Parse header
        if mmap.len() < HEADER_SIZE {
            return Err(BurnpackError::InvalidHeader);
        }

        let header = BurnpackHeader::from_bytes(&mmap[..HEADER_SIZE])?;

        // Verify magic number and version
        if header.magic != MAGIC_NUMBER {
            return Err(BurnpackError::InvalidMagicNumber);
        }

        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Validate metadata size against security limit
        if header.metadata_size > MAX_METADATA_SIZE {
            return Err(BurnpackError::ValidationError(format!(
                "Metadata size {} exceeds maximum allowed size of {} bytes (potential DoS attack)",
                header.metadata_size, MAX_METADATA_SIZE
            )));
        }

        // Parse metadata
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start
            .checked_add(header.metadata_size as usize)
            .ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Metadata size overflow: {} + {}",
                    metadata_start, header.metadata_size
                ))
            })?;

        if mmap.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata = ciborium::de::from_reader_with_recursion_limit(
            &mmap[metadata_start..metadata_end],
            MAX_CBOR_RECURSION_DEPTH,
        )
        .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        // Validate tensor count against security limit
        if metadata.tensors.len() > MAX_TENSOR_COUNT {
            return Err(BurnpackError::ValidationError(format!(
                "File contains {} tensors, exceeding maximum of {} (potential DoS attack)",
                metadata.tensors.len(),
                MAX_TENSOR_COUNT
            )));
        }

        // Validate total file size - ensure file is large enough for all claimed tensor data
        if !metadata.tensors.is_empty() {
            let max_data_offset = metadata
                .tensors
                .values()
                .map(|t| t.data_offsets.1)
                .max()
                .unwrap_or(0);

            let max_data_offset_usize: usize = max_data_offset.try_into().map_err(|_| {
                BurnpackError::ValidationError(format!(
                    "Data offset {} exceeds platform maximum",
                    max_data_offset
                ))
            })?;

            let min_file_size =
                metadata_end
                    .checked_add(max_data_offset_usize)
                    .ok_or_else(|| {
                        BurnpackError::ValidationError("File size calculation overflow".into())
                    })?;

            if mmap.len() < min_file_size {
                return Err(BurnpackError::ValidationError(format!(
                    "File truncated: expected at least {} bytes, got {} bytes",
                    min_file_size,
                    mmap.len()
                )));
            }
        }

        // Convert mmap to bytes::Bytes for zero-copy slicing support
        // bytes::Bytes::from_owner takes ownership and enables efficient slicing
        let shared_bytes = bytes::Bytes::from_owner(mmap);
        let bytes = Bytes::from_shared(shared_bytes, burn_tensor::AllocationProperty::File);

        Ok(Self {
            metadata,
            storage: StorageBackend::Memory(Rc::new(bytes)),
            data_offset: aligned_data_section_start(header.metadata_size as usize),
        })
    }

    /// Load from file - automatically uses memory mapping if available, otherwise uses buffered reading
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        #[cfg(feature = "memmap")]
        {
            // Use memory mapping for efficient access
            Self::from_file_mmap(path)
        }
        #[cfg(not(feature = "memmap"))]
        {
            // Fall back to buffered reading for memory efficiency
            Self::from_file_buffered(path)
        }
    }

    /// Load from file with buffered reading (memory efficient but slower)
    /// This is less efficient than memory mapping but works everywhere
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    pub(crate) fn from_file_buffered<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        let mut file = File::open(&path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

        // Validate maximum file size to prevent resource exhaustion
        let file_size = file
            .metadata()
            .map_err(|e| BurnpackError::IoError(e.to_string()))?
            .len();

        if file_size > MAX_FILE_SIZE {
            return Err(BurnpackError::ValidationError(format!(
                "File size {} bytes exceeds maximum allowed size of {} bytes",
                file_size, MAX_FILE_SIZE
            )));
        }

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let header = BurnpackHeader::from_bytes(&header_bytes)?;

        // Verify version
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Validate metadata size against security limit
        if header.metadata_size > MAX_METADATA_SIZE {
            return Err(BurnpackError::ValidationError(format!(
                "Metadata size {} exceeds maximum allowed size of {} bytes (potential DoS attack)",
                header.metadata_size, MAX_METADATA_SIZE
            )));
        }

        // Read metadata
        let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
        file.read_exact(&mut metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let metadata: BurnpackMetadata = ciborium::de::from_reader_with_recursion_limit(
            metadata_bytes.as_slice(),
            MAX_CBOR_RECURSION_DEPTH,
        )
        .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        // Validate tensor count against security limit
        if metadata.tensors.len() > MAX_TENSOR_COUNT {
            return Err(BurnpackError::ValidationError(format!(
                "File contains {} tensors, exceeding maximum of {} (potential DoS attack)",
                metadata.tensors.len(),
                MAX_TENSOR_COUNT
            )));
        }

        // Calculate metadata end offset
        let metadata_end = HEADER_SIZE
            .checked_add(header.metadata_size as usize)
            .ok_or_else(|| {
                BurnpackError::IoError(format!(
                    "Metadata size overflow: {} + {}",
                    HEADER_SIZE, header.metadata_size
                ))
            })?;

        // Validate total file size - ensure file is large enough for all claimed tensor data
        if !metadata.tensors.is_empty() {
            let max_data_offset = metadata
                .tensors
                .values()
                .map(|t| t.data_offsets.1)
                .max()
                .unwrap_or(0);

            let max_data_offset_usize: usize = max_data_offset.try_into().map_err(|_| {
                BurnpackError::ValidationError(format!(
                    "Data offset {} exceeds platform maximum",
                    max_data_offset
                ))
            })?;

            let min_file_size =
                metadata_end
                    .checked_add(max_data_offset_usize)
                    .ok_or_else(|| {
                        BurnpackError::ValidationError("File size calculation overflow".into())
                    })?;

            // Get actual file size
            let file_size = file
                .metadata()
                .map_err(|e| BurnpackError::IoError(e.to_string()))?
                .len() as usize;

            if file_size < min_file_size {
                return Err(BurnpackError::ValidationError(format!(
                    "File truncated: expected at least {} bytes, got {} bytes",
                    min_file_size, file_size
                )));
            }
        }

        Ok(Self {
            metadata,
            storage: StorageBackend::FileBuffered {
                file: Rc::new(RefCell::new(file)),
            },
            data_offset: aligned_data_section_start(header.metadata_size as usize),
        })
    }

    /// Get all tensor snapshots at once for efficient loading (always copies data)
    pub fn get_snapshots(&self) -> Result<Vec<TensorSnapshot>, BurnpackError> {
        self.get_snapshots_internal(false)
    }

    /// Get all tensor snapshots with optional zero-copy loading.
    ///
    /// When `zero_copy` is true and the backend supports it (Memory backend with
    /// `Bytes::from_shared()`), tensor data is sliced without copying. This keeps
    /// the original data alive as long as any tensor holds a reference.
    ///
    /// When `zero_copy` is false or the backend doesn't support it, data is copied
    /// into newly allocated buffers (default behavior).
    pub fn get_snapshots_zero_copy(
        &self,
        zero_copy: bool,
    ) -> Result<Vec<TensorSnapshot>, BurnpackError> {
        self.get_snapshots_internal(zero_copy)
    }

    /// Internal implementation with optional zero-copy support
    fn get_snapshots_internal(
        &self,
        zero_copy: bool,
    ) -> Result<Vec<TensorSnapshot>, BurnpackError> {
        let mut snapshots = Vec::new();

        for (name, descriptor) in &self.metadata.tensors {
            // Clone metadata for use in closure
            // Convert shape dimensions with overflow checking
            let shape: Shape = Shape::from(descriptor
                .shape
                .iter()
                .map(|&s| {
                    s.try_into().map_err(|_| {
                        BurnpackError::ValidationError(format!(
                            "Tensor '{}' has corrupted shape data: dimension {} exceeds platform maximum",
                            name, s
                        ))
                    })
                })
                .collect::<Result<Vec<usize>, BurnpackError>>()?);

            let dtype = descriptor.dtype;

            // Clone storage reference for the closure
            let storage = match &self.storage {
                StorageBackend::Memory(data) => StorageBackend::Memory(data.clone()),
                #[cfg(feature = "std")]
                StorageBackend::FileBuffered { file } => {
                    StorageBackend::FileBuffered { file: file.clone() }
                }
            };

            // Always use absolute positions for all backends
            // Convert offsets with overflow checking
            let offset_start: usize = descriptor.data_offsets.0.try_into().map_err(|_| {
                BurnpackError::ValidationError(format!(
                    "Tensor '{}' has corrupted offset data: start offset {} exceeds platform maximum",
                    name, descriptor.data_offsets.0
                ))
            })?;

            let offset_end: usize = descriptor.data_offsets.1.try_into().map_err(|_| {
                BurnpackError::ValidationError(format!(
                    "Tensor '{}' has corrupted offset data: end offset {} exceeds platform maximum",
                    name, descriptor.data_offsets.1
                ))
            })?;

            let start = self.data_offset.checked_add(offset_start).ok_or_else(|| {
                BurnpackError::ValidationError(format!(
                    "Tensor '{}' has corrupted offset data: start offset overflow {} + {}",
                    name, self.data_offset, offset_start
                ))
            })?;

            let end = self.data_offset.checked_add(offset_end).ok_or_else(|| {
                BurnpackError::ValidationError(format!(
                    "Tensor '{}' has corrupted offset data: end offset overflow {} + {}",
                    name, self.data_offset, offset_end
                ))
            })?;

            // Clone shape for the closure (TensorSnapshot::from_closure will also need it)
            let shape_for_closure = shape.clone();

            // Validate offset range
            if end < start {
                return Err(BurnpackError::ValidationError(format!(
                    "Tensor '{}' has corrupted offset data: end offset {} < start offset {}",
                    name, end, start
                )));
            }

            // Validate tensor size against security limit
            let tensor_size = end - start;
            if tensor_size > MAX_TENSOR_SIZE {
                return Err(BurnpackError::ValidationError(format!(
                    "Tensor '{}' size {} exceeds maximum allowed size of {} bytes (potential DoS attack)",
                    name, tensor_size, MAX_TENSOR_SIZE
                )));
            }

            // Restore param_id if it was saved, otherwise generate
            let tensor_id = descriptor
                .param_id
                .map(ParamId::from)
                .unwrap_or_else(ParamId::new);

            // Create the data-loading closure based on zero_copy flag
            let data_fn: Rc<dyn Fn() -> Result<TensorData, crate::TensorSnapshotError>> =
                if zero_copy {
                    // Zero-copy closure: slice without copying, error if not supported
                    Rc::new(move || {
                        let bytes = storage.slice_bytes(start, end).map_err(|e| {
                            crate::TensorSnapshotError::IoError(format!(
                                "Zero-copy slice failed: {}",
                                e
                            ))
                        })?;
                        Ok(TensorData::from_bytes(
                            bytes,
                            shape_for_closure.clone(),
                            dtype,
                        ))
                    })
                } else {
                    // Copying closure: always allocate and copy
                    Rc::new(move || {
                        let len = end - start;
                        // TODO Should be allocated by the backend in the future
                        // See https://github.com/tracel-ai/burn/pull/3792#discussion_r2416812091
                        let mut data_bytes = vec![0u8; len];
                        storage.read_into(&mut data_bytes, start).map_err(|e| {
                            crate::TensorSnapshotError::IoError(format!(
                                "Failed to read tensor data: {}",
                                e
                            ))
                        })?;
                        Ok(TensorData::from_bytes_vec(
                            data_bytes,
                            shape_for_closure.clone(),
                            dtype,
                        ))
                    })
                };

            // Create lazy TensorSnapshot
            let snapshot = TensorSnapshot::from_closure(
                data_fn,
                dtype,
                shape,
                name.split('.').map(|s| s.to_string()).collect(),
                vec![],    // empty container_stack
                tensor_id, // restored or newly generated param id
            );

            snapshots.push(snapshot);
        }

        Ok(snapshots)
    }

    // Legacy methods for test compatibility - will be removed

    /// Get tensor as TensorSnapshot with lazy loading
    #[allow(dead_code)]
    pub(crate) fn get_tensor_snapshot(&self, name: &str) -> Result<TensorSnapshot, BurnpackError> {
        let snapshots = self.get_snapshots()?;
        snapshots
            .into_iter()
            .find(|s| s.full_path() == name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))
    }

    /// Get list of tensor names
    #[allow(dead_code)]
    pub(crate) fn tensor_names(&self) -> Vec<&str> {
        self.metadata
            .tensors
            .keys()
            .map(|name| name.as_str())
            .collect()
    }

    /// Get metadata
    #[allow(dead_code)]
    pub(crate) fn metadata(&self) -> &BurnpackMetadata {
        &self.metadata
    }

    /// Get tensor data as raw bytes
    #[allow(dead_code)]
    pub(crate) fn get_tensor_data(&self, name: &str) -> Result<Vec<u8>, BurnpackError> {
        let descriptor = self
            .metadata
            .tensors
            .get(name)
            .ok_or_else(|| BurnpackError::TensorNotFound(name.to_string()))?;

        // Always use absolute positions for all backends
        // Convert offsets with overflow checking
        let offset_start: usize = descriptor.data_offsets.0.try_into().map_err(|_| {
            BurnpackError::IoError(format!(
                "Tensor '{}' has corrupted offset data: start offset {} exceeds platform maximum",
                name, descriptor.data_offsets.0
            ))
        })?;

        let offset_end: usize = descriptor.data_offsets.1.try_into().map_err(|_| {
            BurnpackError::IoError(format!(
                "Tensor '{}' has corrupted offset data: end offset {} exceeds platform maximum",
                name, descriptor.data_offsets.1
            ))
        })?;

        let start = self.data_offset.checked_add(offset_start).ok_or_else(|| {
            BurnpackError::IoError(format!(
                "Tensor '{}' has corrupted offset data: start offset overflow {} + {}",
                name, self.data_offset, offset_start
            ))
        })?;

        let end = self.data_offset.checked_add(offset_end).ok_or_else(|| {
            BurnpackError::IoError(format!(
                "Tensor '{}' has corrupted offset data: end offset overflow {} + {}",
                name, self.data_offset, offset_end
            ))
        })?;

        // Validate offset range
        if end < start {
            return Err(BurnpackError::IoError(format!(
                "Tensor '{}' has corrupted offset data: end offset {} < start offset {}",
                name, end, start
            )));
        }

        let len = end - start;
        let mut buffer = vec![0u8; len];
        self.storage.read_into(&mut buffer, start)?;
        Ok(buffer)
    }
}
