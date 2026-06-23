use super::base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAGIC_NUMBER, Metadata, Scalar, TENSOR_ALIGNMENT,
    TensorDescriptor, aligned_data_section_start,
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
use std::io::{Read, Write};
#[cfg(feature = "std")]
use std::path::Path;

/// Align an offset to the specified alignment boundary.
///
/// Returns the smallest value >= `offset` that is a multiple of `alignment`.
#[inline]
const fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset.div_ceil(alignment) * alignment
}

/// Maximum number of bytes materialized from a single tensor at a time while
/// streaming its data into a [`Sink`].
///
/// Large device-resident tensors are read back to host memory lazily, one
/// [`Bytes::view`] window at a time, instead of all at once. This keeps the
/// transient (often pinned) host staging buffer bounded by this size regardless
/// of how large the tensor is. The value is a multiple of [`TENSOR_ALIGNMENT`]
/// so each window starts on an aligned device offset.
const WRITE_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Writer for creating Burnpack files
pub struct Writer {
    /// Tensors to write
    pub(crate) tensors: Vec<Tensor>,
    /// Metadata key-value pairs
    pub(crate) metadata: BTreeMap<String, String>,
    /// Typed scalars keyed by name
    pub(crate) scalars: BTreeMap<String, Scalar>,
}

impl Writer {
    /// Create a new writer
    pub fn new(tensors: Vec<Tensor>) -> Self {
        Self {
            tensors,
            metadata: BTreeMap::new(),
            scalars: BTreeMap::new(),
        }
    }

    /// Builder pattern: add metadata and return self
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Builder pattern: add a typed scalar and return self.
    pub fn with_scalar(mut self, key: &str, value: Scalar) -> Self {
        self.scalars.insert(key.to_string(), value);
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
    pub fn write_into(self, buffer: &mut [u8]) -> Result<(), Error> {
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
    pub fn into_bytes(self) -> Result<Bytes, Error> {
        let layout = self.plan()?;
        let mut buffer = vec![0u8; layout.total_size()];

        let mut sink = BufferSink {
            buffer: &mut buffer,
            offset: 0,
        };
        self.write_container(&layout, &mut sink)?;

        Ok(Bytes::from_bytes_vec(buffer))
    }

    /// Stream the container to any [`std::io::Write`] without materializing the whole image
    /// first. Streaming counterpart to [`into_bytes`](Self::into_bytes); pairs with
    /// [`Reader::from_reader`](crate::Reader::from_reader). The writer is flushed before returning.
    #[cfg(feature = "std")]
    pub fn write_to<W: Write>(self, writer: W) -> Result<(), Error> {
        let layout = self.plan()?;
        let mut sink = WriterSink { writer };
        self.write_container(&layout, &mut sink)?;
        sink.writer
            .flush()
            .map_err(|e| Error::IoError(e.to_string()))
    }

    /// Write directly to a file (more memory efficient for large models).
    ///
    /// If `path` has no extension, the canonical [`crate::EXTENSION`] (`.bpk`) is appended.
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<(), Error> {
        let path = path.as_ref();
        let path = if path.extension().is_none() {
            path.with_extension(crate::EXTENSION)
        } else {
            path.to_path_buf()
        };

        let file = File::create(path).map_err(|e| Error::IoError(e.to_string()))?;
        self.write_to(file)
    }

    /// Build the complete on-disk layout: header, serialized metadata, and the
    /// position and size of the (aligned) tensor data section.
    fn plan(&self) -> Result<Layout, Error> {
        let (metadata, metadata_bytes, data_size) = self.build_metadata()?;

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

        Ok(Layout {
            metadata,
            metadata_bytes,
            header,
            data_section_start,
            data_size,
        })
    }

    /// Serialize the metadata structure (tensor descriptors + key-value pairs) to CBOR.
    ///
    /// Also returns the size of the tensor data section, computed while assigning offsets.
    fn build_metadata(&self) -> Result<(Metadata, Vec<u8>, usize), Error> {
        let (tensors, data_size) = self.build_descriptors()?;
        let metadata = Metadata {
            tensors,
            metadata: self.metadata.clone(),
            scalars: self.scalars.clone(),
        };

        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| Error::MetadataSerializationError(e.to_string()))?;

        Ok((metadata, metadata_bytes, data_size))
    }

    /// Build tensor descriptors, assigning each tensor an aligned offset within
    /// the data section so that absolute file positions are mmap-friendly.
    ///
    /// Returns the descriptors plus the total data-section size — the running offset after the
    /// last tensor. Offsets only grow, so this is also the highest descriptor end offset.
    fn build_descriptors(&self) -> Result<(BTreeMap<String, TensorDescriptor>, usize), Error> {
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

            // Descriptors are keyed by name, but the tensor data is written from the
            // (ordered) `self.tensors` list. A duplicate name would collapse to a single
            // descriptor while still writing two data blocks, corrupting the container.
            if tensors
                .insert(
                    tensor.name.clone(),
                    TensorDescriptor {
                        dtype: tensor.dtype,
                        shape: tensor.shape.iter().map(|&s| s as u64).collect(),
                        data_offsets: (aligned_start, end),
                        param_id: tensor.param_id,
                    },
                )
                .is_some()
            {
                return Err(Error::ValidationError(format!(
                    "Duplicate tensor name '{}'",
                    tensor.name
                )));
            }

            current_offset = end;
        }

        Ok((tensors, current_offset as usize))
    }

    /// Emit the full container — header, metadata, alignment padding, then tensor data
    /// — into `sink`, which decides where the bytes ultimately land.
    fn write_container(self, layout: &Layout, sink: &mut impl Sink) -> Result<(), Error> {
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
    fn write_tensors(self, metadata: &Metadata, sink: &mut impl Sink) -> Result<(), Error> {
        // Position within the data section (relative to its aligned start).
        let mut data_offset = 0usize;

        for tensor in self.tensors.into_iter() {
            let (aligned_offset, data) = Self::resolve_tensor(tensor, metadata)?;

            if aligned_offset > data_offset {
                sink.pad(aligned_offset - data_offset)?;
                data_offset = aligned_offset;
            }

            Self::write_tensor_data(&data, sink)?;
            data_offset += data.len();
        }

        Ok(())
    }

    /// Stream a single tensor's bytes into `sink`, materializing at most
    /// [`WRITE_CHUNK_SIZE`] bytes at a time.
    ///
    /// When the backing supports zero-copy windows — device-resident
    /// ([lazy](burn_std::Bytes) device readback), file, or shared buffers — each
    /// chunk is taken as a [`Bytes::view`] and read just-in-time, then dropped
    /// before the next one. A large device tensor is therefore copied to host in
    /// bounded pieces rather than through one big (pinned) staging buffer, so the
    /// whole tensor never has to be resident at once.
    ///
    /// Backings without a zero-copy window (e.g. a plain heap `Vec`) are already
    /// host-resident, so [`Bytes::view`] reports it can't window them and the
    /// remaining bytes are written in a single pass.
    fn write_tensor_data(data: &Bytes, sink: &mut impl Sink) -> Result<(), Error> {
        let len = data.len();
        let mut offset = 0;

        while offset < len {
            let end = (offset + WRITE_CHUNK_SIZE).min(len);
            match data.view(offset, end) {
                Ok(chunk) => {
                    sink.write(&chunk)?;
                    offset = end;
                }
                // No zero-copy window available (already host-resident): write
                // whatever remains in one shot. View support is a property of the
                // backing, so this only ever happens on the first iteration.
                Err(_) => {
                    sink.write(&data[offset..])?;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Look up a tensor's aligned offset from the metadata and validate that its
    /// bytes match the length the descriptor reserved for it.
    fn resolve_tensor(tensor: Tensor, metadata: &Metadata) -> Result<(usize, Bytes), Error> {
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
            return Err(Error::TensorBytesSizeMismatch(format!(
                "tensor '{}' has inconsistent length (expected {}, got {})",
                tensor.name, declared_len, actual_len
            )));
        }

        Ok((start as usize, tensor.bytes))
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

/// Sink that streams to any [`std::io::Write`].
#[cfg(feature = "std")]
struct WriterSink<W: Write> {
    writer: W,
}

#[cfg(feature = "std")]
impl<W: Write> Sink for WriterSink<W> {
    fn pad(&mut self, count: usize) -> Result<(), Error> {
        // Stream zeros without allocating a `count`-sized buffer per call.
        std::io::copy(&mut std::io::repeat(0).take(count as u64), &mut self.writer)
            .map(|_| ())
            .map_err(|e| Error::IoError(e.to_string()))
    }

    fn write(&mut self, data: &[u8]) -> Result<(), Error> {
        self.writer
            .write_all(data)
            .map_err(|e| Error::IoError(e.to_string()))
    }
}
