use super::base::{
    BurnpackError, BurnpackHeader, BurnpackMetadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
};
use crate::TensorSnapshot;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use burn_core::module::ParamId;
use burn_tensor::{Bytes, TensorData};

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
    /// Memory-based storage
    Memory(Rc<Bytes>),
    /// Memory-mapped file storage (efficient for large files)
    #[cfg(all(feature = "std", feature = "memmap"))]
    Mmap(Rc<memmap2::Mmap>),
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
    /// # Notes
    /// The caller allocates the buffer, which allows for buffer reuse and future optimizations
    /// like memory pools and pinned memory.
    pub(crate) fn read_into(&self, bytes: &mut [u8], offset: usize) -> Result<(), BurnpackError> {
        match self {
            StorageBackend::Memory(data) => {
                let data_bytes = data.as_ref();
                let end = offset.saturating_add(bytes.len()).min(data_bytes.len());
                let safe_offset = offset.min(data_bytes.len());
                let available = &data_bytes[safe_offset..end];
                bytes[..available.len()].copy_from_slice(available);
                Ok(())
            }
            #[cfg(all(feature = "std", feature = "memmap"))]
            StorageBackend::Mmap(mmap) => {
                let mmap_bytes = mmap.as_ref();
                let end = offset.saturating_add(bytes.len()).min(mmap_bytes.len());
                let safe_offset = offset.min(mmap_bytes.len());
                let available = &mmap_bytes[safe_offset..end];
                bytes[..available.len()].copy_from_slice(available);
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
            #[cfg(all(feature = "std", feature = "memmap"))]
            StorageBackend::Mmap(mmap) => Ok(mmap.as_ref()),
            #[cfg(feature = "std")]
            StorageBackend::FileBuffered { .. } => Err(BurnpackError::IoError(
                "Cannot get full bytes reference for FileBuffered backend".into(),
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

        // Parse metadata
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if bytes.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata =
            ciborium::de::from_reader(&bytes[metadata_start..metadata_end])
                .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        Ok(Self {
            metadata,
            storage: StorageBackend::Memory(Rc::new(bytes)),
            data_offset: metadata_end,
        })
    }

    /// Load from file with memory mapping (most efficient for large files)
    #[cfg(all(feature = "std", feature = "memmap"))]
    pub(crate) fn from_file_mmap<P: AsRef<Path>>(path: P) -> Result<Self, BurnpackError> {
        let file = File::open(&path).map_err(|e| BurnpackError::IoError(e.to_string()))?;

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

        // Parse metadata
        let metadata_start = HEADER_SIZE;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if mmap.len() < metadata_end {
            return Err(BurnpackError::InvalidHeader);
        }

        let metadata: BurnpackMetadata =
            ciborium::de::from_reader(&mmap[metadata_start..metadata_end])
                .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        Ok(Self {
            metadata,
            storage: StorageBackend::Mmap(Rc::new(mmap)),
            data_offset: metadata_end,
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

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let header = BurnpackHeader::from_bytes(&header_bytes)?;

        // Verify version
        if header.version > FORMAT_VERSION {
            return Err(BurnpackError::InvalidVersion);
        }

        // Read metadata
        let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
        file.read_exact(&mut metadata_bytes)
            .map_err(|e| BurnpackError::IoError(e.to_string()))?;

        let metadata: BurnpackMetadata = ciborium::de::from_reader(metadata_bytes.as_slice())
            .map_err(|e| BurnpackError::MetadataDeserializationError(e.to_string()))?;

        // Store file handle for reuse
        let metadata_end = HEADER_SIZE + header.metadata_size as usize;

        Ok(Self {
            metadata,
            storage: StorageBackend::FileBuffered {
                file: Rc::new(RefCell::new(file)),
            },
            data_offset: metadata_end,
        })
    }

    /// Get all tensor snapshots at once for efficient loading
    pub fn get_snapshots(&self) -> Vec<TensorSnapshot> {
        let mut snapshots = Vec::new();

        for (name, descriptor) in &self.metadata.tensors {
            // Clone metadata for use in closure
            let shape: Vec<usize> = descriptor.shape.iter().map(|&s| s as usize).collect();
            let dtype = descriptor.dtype;

            // Clone storage reference for the closure
            let storage = match &self.storage {
                StorageBackend::Memory(data) => StorageBackend::Memory(data.clone()),
                #[cfg(all(feature = "std", feature = "memmap"))]
                StorageBackend::Mmap(mmap) => StorageBackend::Mmap(mmap.clone()),
                #[cfg(feature = "std")]
                StorageBackend::FileBuffered { file } => {
                    StorageBackend::FileBuffered { file: file.clone() }
                }
            };

            // Always use absolute positions for all backends
            let start = self.data_offset + descriptor.data_offsets.0 as usize;
            let end = self.data_offset + descriptor.data_offsets.1 as usize;

            // Clone shape for the closure (TensorSnapshot::from_closure will also need it)
            let shape_for_closure = shape.clone();

            // Create lazy TensorSnapshot
            let snapshot = TensorSnapshot::from_closure(
                Rc::new(move || {
                    // This closure is only called when data is actually needed
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
                }),
                dtype,
                shape,
                name.split('.').map(|s| s.to_string()).collect(),
                vec![],         // empty container_stack
                ParamId::new(), // new unique id
            );

            snapshots.push(snapshot);
        }

        snapshots
    }

    // Legacy methods for test compatibility - will be removed

    /// Get tensor as TensorSnapshot with lazy loading
    #[allow(dead_code)]
    pub(crate) fn get_tensor_snapshot(&self, name: &str) -> Result<TensorSnapshot, BurnpackError> {
        let snapshots = self.get_snapshots();
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
        let start = self.data_offset + descriptor.data_offsets.0 as usize;
        let end = self.data_offset + descriptor.data_offsets.1 as usize;
        let len = end - start;
        let mut buffer = vec![0u8; len];
        self.storage.read_into(&mut buffer, start)?;
        Ok(buffer)
    }
}
