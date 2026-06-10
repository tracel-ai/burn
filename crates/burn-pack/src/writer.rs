use super::base::{
    Error, Header, Metadata, FORMAT_VERSION, HEADER_SIZE, MAGIC_NUMBER,
    TENSOR_ALIGNMENT, TensorDescriptor, aligned_data_section_start,
};
use super::tensor::Tensor;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use burn_std::Bytes;

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
pub struct Writer {
    /// Tensors to write
    pub(crate) tensors: Vec<Tensor>,
    /// Metadata key-value pairs
    pub(crate) metadata: BTreeMap<String, String>,
}

impl Writer {
    /// Create a new writer
    pub fn new(tensors: Vec<Tensor>) -> Self {
        Self {
            tensors,
            metadata: BTreeMap::new(),
        }
    }

    /// Builder pattern: add metadata and return self
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Calculate the total size needed for the burnpack data.
    ///
    /// This is useful when you want to pre-allocate a buffer for `write_into()`.
    /// The size includes padding bytes for both metadata alignment and tensor alignment.
    pub fn size(&self) -> Result<usize, Error> {
        Ok(self.plan()?.total_size())
    }

    /// Write burnpack data into a caller-provided buffer.
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
    pub fn write_into(&self, buffer: &mut [u8]) -> Result<(), Error> {
        let layout = self.plan()?;
        let total_size = layout.total_size();

        if buffer.len() < total_size {
            return Err(Error::IoError(format!(
                "Buffer too small: need {} bytes, got {} bytes",
                total_size,
                buffer.len()
            )));
        }

        let mut sink = BufferSink { buffer, offset: 0 };
        self.write_container(&layout, &mut sink)
    }

    /// Write to a byte buffer (convenience method).
    ///
    /// This allocates a buffer internally and writes the burnpack data.
    /// For more control over buffer allocation, use `size()` + `write_into()`.
    pub fn to_bytes(&self) -> Result<Bytes, Error> {
        let layout = self.plan()?;
        let mut buffer = vec![0u8; layout.total_size()];

        let mut sink = BufferSink {
            buffer: &mut buffer,
            offset: 0,
        };
        self.write_container(&layout, &mut sink)?;

        Ok(Bytes::from_bytes_vec(buffer))
    }

    /// Write directly to a file (more memory efficient for large models).
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let layout = self.plan()?;
        let file = File::create(path).map_err(|e| Error::IoError(e.to_string()))?;

        let mut sink = FileSink { file };
        self.write_container(&layout, &mut sink)?;

        sink.file.flush().map_err(|e| Error::IoError(e.to_string()))
    }

    /// Build the complete on-disk layout: header, serialized metadata, and the
    /// position and size of the (aligned) tensor data section.
    fn plan(&self) -> Result<Layout, Error> {
        let (metadata, metadata_bytes) = self.build_metadata()?;

        let metadata_size: u32 = metadata_bytes.len().try_into().map_err(|_| {
            Error::IoError(format!(
                "Metadata size {} exceeds maximum of {} bytes",
                metadata_bytes.len(),
                u32::MAX
            ))
        })?;

        let header = Header {
            magic: MAGIC_NUMBER,
            version: FORMAT_VERSION,
            metadata_size,
        };

        let data_section_start = aligned_data_section_start(metadata_bytes.len());
        let data_size = Self::data_section_size(&metadata);

        Ok(Layout {
            metadata,
            metadata_bytes,
            header,
            data_section_start,
            data_size,
        })
    }

    /// Serialize the metadata structure (tensor descriptors + key-value pairs) to CBOR.
    fn build_metadata(&self) -> Result<(Metadata, Vec<u8>), Error> {
        let metadata = Metadata {
            tensors: self.build_descriptors()?,
            metadata: self.metadata.clone(),
        };

        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| Error::IoError(e.to_string()))?;

        Ok((metadata, metadata_bytes))
    }

    /// Build tensor descriptors, assigning each tensor an aligned offset within
    /// the data section so that absolute file positions are mmap-friendly.
    fn build_descriptors(&self) -> Result<BTreeMap<String, TensorDescriptor>, Error> {
        let mut tensors = BTreeMap::new();
        let mut current_offset = 0u64;

        for tensor in &self.tensors {
            let data_len = tensor.bytes.len() as u64;

            // Align the start offset for mmap zero-copy support.
            let aligned_start = align_offset(current_offset, TENSOR_ALIGNMENT);
            let end = aligned_start.checked_add(data_len).ok_or_else(|| {
                Error::IoError(format!(
                    "Tensor offset overflow: {} + {} exceeds maximum",
                    aligned_start, data_len
                ))
            })?;

            tensors.insert(
                tensor.name.clone(),
                TensorDescriptor {
                    dtype: tensor.dtype,
                    shape: tensor.shape.iter().map(|&s| s as u64).collect(),
                    data_offsets: (aligned_start, end),
                    param_id: tensor.param_id,
                },
            );

            current_offset = end;
        }

        Ok(tensors)
    }

    /// Size of the tensor data section, derived from the highest descriptor end offset.
    fn data_section_size(metadata: &Metadata) -> usize {
        metadata
            .tensors
            .values()
            .map(|t| t.data_offsets.1)
            .max()
            .unwrap_or(0) as usize
    }

    /// Emit the full container — header, metadata, alignment padding, then tensor data
    /// — into `sink`, which decides where the bytes ultimately land.
    fn write_container(&self, layout: &Layout, sink: &mut impl Sink) -> Result<(), Error> {
        sink.write(&layout.header.into_bytes())?;
        sink.write(&layout.metadata_bytes)?;

        // Pad so the data section starts at its aligned position.
        let unaligned_data_start = HEADER_SIZE + layout.metadata_bytes.len();
        if layout.data_section_start > unaligned_data_start {
            sink.pad(layout.data_section_start - unaligned_data_start)?;
        }

        self.write_tensors(&layout.metadata, sink)
    }

    /// Write each tensor's data into `sink`, inserting alignment padding between
    /// tensors so every tensor lands at its descriptor's aligned offset.
    fn write_tensors(&self, metadata: &Metadata, sink: &mut impl Sink) -> Result<(), Error> {
        // Position within the data section (relative to its aligned start).
        let mut data_offset = 0usize;

        for tensor in &self.tensors {
            let (aligned_offset, data) = Self::resolve_tensor(tensor, metadata)?;

            if aligned_offset > data_offset {
                sink.pad(aligned_offset - data_offset)?;
                data_offset = aligned_offset;
            }

            sink.write(data)?;
            data_offset += data.len();
        }

        Ok(())
    }

    /// Look up a tensor's aligned offset from the metadata and validate that its
    /// bytes match the length the descriptor reserved for it.
    fn resolve_tensor<'t>(
        tensor: &'t Tensor,
        metadata: &Metadata,
    ) -> Result<(usize, &'t [u8]), Error> {
        let descriptor = metadata.tensors.get(&tensor.name).ok_or_else(|| {
            Error::IoError(format!(
                "Internal error: tensor '{}' not found in metadata",
                tensor.name
            ))
        })?;

        let (start, end) = descriptor.data_offsets;
        let declared_len = (end - start) as usize;
        let actual_len = tensor.bytes.len();
        if actual_len != declared_len {
            return Err(Error::IoError(format!(
                "Data corruption: tensor '{}' has inconsistent length (expected {}, got {})",
                tensor.name, declared_len, actual_len
            )));
        }

        Ok((start as usize, &tensor.bytes))
    }
}

/// The computed on-disk layout of a burnpack container.
///
/// Captures everything needed to emit the bytes: the serialized metadata, the
/// header, where the aligned data section begins, and how large it is. Built once
/// via [`Writer::plan`] and shared by `size`, `write_into`, `to_bytes`, and
/// `write_to_file`.
struct Layout {
    metadata: Metadata,
    metadata_bytes: Vec<u8>,
    header: Header,
    data_section_start: usize,
    data_size: usize,
}

impl Layout {
    /// Total number of bytes the container occupies.
    fn total_size(&self) -> usize {
        self.data_section_start + self.data_size
    }
}

/// A sequential destination for the bytes of a burnpack container.
///
/// Padding and data are written in order; each implementation advances its own
/// cursor, letting the writer stay agnostic about whether bytes land in a buffer
/// or a file.
trait Sink {
    /// Write `count` zero bytes of alignment padding.
    fn pad(&mut self, count: usize) -> Result<(), Error>;
    /// Write `data` verbatim.
    fn write(&mut self, data: &[u8]) -> Result<(), Error>;
}

/// Sink that copies into a caller-provided buffer.
struct BufferSink<'a> {
    buffer: &'a mut [u8],
    offset: usize,
}

impl Sink for BufferSink<'_> {
    fn pad(&mut self, count: usize) -> Result<(), Error> {
        self.buffer[self.offset..self.offset + count].fill(0);
        self.offset += count;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<(), Error> {
        self.buffer[self.offset..self.offset + data.len()].copy_from_slice(data);
        self.offset += data.len();
        Ok(())
    }
}

/// Sink that streams directly to a file.
#[cfg(feature = "std")]
struct FileSink {
    file: File,
}

#[cfg(feature = "std")]
impl Sink for FileSink {
    fn pad(&mut self, count: usize) -> Result<(), Error> {
        self.file
            .write_all(&vec![0u8; count])
            .map_err(|e| Error::IoError(e.to_string()))
    }

    fn write(&mut self, data: &[u8]) -> Result<(), Error> {
        self.file
            .write_all(data)
            .map_err(|e| Error::IoError(e.to_string()))
    }
}
