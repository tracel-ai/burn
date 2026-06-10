use super::base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAX_CBOR_RECURSION_DEPTH, MAX_METADATA_SIZE,
    MAX_TENSOR_COUNT, MAX_TENSOR_SIZE, Metadata, TensorDescriptor, aligned_data_section_start,
};
use super::tensor::Tensor;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use burn_std::{Bytes, Shape};

#[cfg(feature = "std")]
use super::base::MAX_FILE_SIZE;
#[cfg(feature = "std")]
use alloc::vec;
#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::io::Read;
#[cfg(feature = "std")]
use std::path::{Path, PathBuf};

/// Where a [`Reader`] gets its tensor data from.
enum Source {
    /// The whole pack lives in memory.
    Memory(Bytes),
    /// The pack lives in a file; tensor data is read lazily via [`Bytes::from_file`].
    #[cfg(feature = "std")]
    File(PathBuf),
}

/// Reader for loading burnpack containers.
pub struct Reader {
    metadata: Metadata,
    source: Source,
    /// Absolute byte offset where the (256-byte aligned) tensor data section starts.
    data_offset: usize,
}

impl Reader {
    /// Load a pack from an in-memory [`Bytes`] buffer.
    ///
    /// Each tensor's bytes are copied out of the buffer on [`get_tensors`](Self::get_tensors).
    /// For large models, prefer [`from_file`](Self::from_file), which keeps tensor data lazy.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, Error> {
        let header = read_header(&bytes)?;
        let metadata_end = HEADER_SIZE
            .checked_add(header.metadata_size as usize)
            .ok_or(Error::InvalidHeader)?;
        if bytes.len() < metadata_end {
            return Err(Error::InvalidHeader);
        }
        let metadata = parse_metadata(&bytes[HEADER_SIZE..metadata_end])?;

        let available = bytes.len();
        Self::assemble(&header, metadata, Source::Memory(bytes), available)
    }

    /// Load a pack from a file.
    ///
    /// Only the header and metadata are read up front; each tensor's data is backed by
    /// [`Bytes::from_file`] and read lazily when accessed. This integrates with the Burn
    /// ecosystem's file allocation (pinned-memory staging) for fast file-to-GPU transfers.
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut file = File::open(&path).map_err(io_err)?;

        let file_size = file.metadata().map_err(io_err)?.len();
        if file_size > MAX_FILE_SIZE {
            return Err(Error::ValidationError(format!(
                "File size {file_size} bytes exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
            )));
        }

        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes).map_err(io_err)?;
        let header = read_header(&header_bytes)?;

        let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
        file.read_exact(&mut metadata_bytes).map_err(io_err)?;
        let metadata = parse_metadata(&metadata_bytes)?;

        let source = Source::File(path.as_ref().to_path_buf());
        Self::assemble(&header, metadata, source, file_size as usize)
    }

    /// Finish construction once the header, metadata, and data source are known.
    ///
    /// Centralizes the truncation check and the aligned data-section offset so both
    /// [`from_bytes`](Self::from_bytes) and [`from_file`](Self::from_file) stay in sync.
    /// `available` is the number of bytes the source can actually supply.
    fn assemble(
        header: &Header,
        metadata: Metadata,
        source: Source,
        available: usize,
    ) -> Result<Self, Error> {
        let metadata_end = HEADER_SIZE + header.metadata_size as usize;
        validate_total_size(&metadata, metadata_end, available)?;

        Ok(Self {
            metadata,
            source,
            data_offset: aligned_data_section_start(header.metadata_size as usize),
        })
    }

    /// Get all tensors in the pack.
    ///
    /// File-backed tensors are read lazily on access; in-memory tensors are copied out of
    /// the source buffer.
    pub fn get_tensors(&self) -> Result<Vec<Tensor>, Error> {
        let descriptors: Vec<(&String, &TensorDescriptor)> = self.metadata.tensors.iter().collect();
        let ranges = descriptors
            .iter()
            .map(|(name, d)| self.tensor_range(name, d))
            .collect::<Result<Vec<_>, _>>()?;

        let data = self.slice_tensors(&ranges)?;

        descriptors
            .into_iter()
            .zip(data)
            .map(|((name, d), bytes)| make_tensor(name, d, bytes))
            .collect()
    }

    /// The user-supplied key/value metadata stored alongside the tensors.
    ///
    /// For per-tensor info (dtype/shape/param id), use [`get_tensors`](Self::get_tensors) — for a
    /// file-backed reader this does not read any tensor data until a tensor's bytes are accessed.
    pub fn metadata(&self) -> &alloc::collections::BTreeMap<String, String> {
        &self.metadata.metadata
    }

    /// The names of all tensors in the pack, in sorted (alphabetical) order.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.tensors.keys().map(|n| n.as_str()).collect()
    }

    /// Read a single tensor's raw little-endian bytes by name (always copies).
    ///
    /// Returns [`Error::TensorNotFound`] if no tensor with that name exists.
    pub fn tensor_data(&self, name: &str) -> Result<Vec<u8>, Error> {
        let descriptor = self
            .metadata
            .tensors
            .get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        let range = self.tensor_range(name, descriptor)?;
        let mut bytes = self.slice_tensors(&[range])?;
        let bytes = bytes.pop().expect("one range yields one slice");
        let slice: &[u8] = &bytes;
        Ok(slice.to_vec())
    }

    /// Compute and validate the absolute `[start, end)` byte range of a tensor.
    fn tensor_range(
        &self,
        name: &str,
        descriptor: &TensorDescriptor,
    ) -> Result<(usize, usize), Error> {
        let to_usize = |offset: u64| -> Result<usize, Error> {
            offset.try_into().map_err(|_| {
                Error::ValidationError(format!(
                    "Tensor '{name}' has corrupted offset data: offset {offset} exceeds platform maximum"
                ))
            })
        };
        let overflow = || {
            Error::ValidationError(format!(
                "Tensor '{name}' has corrupted offset data: overflow"
            ))
        };

        let start = self
            .data_offset
            .checked_add(to_usize(descriptor.data_offsets.0)?)
            .ok_or_else(overflow)?;
        let end = self
            .data_offset
            .checked_add(to_usize(descriptor.data_offsets.1)?)
            .ok_or_else(overflow)?;

        if end < start {
            return Err(Error::ValidationError(format!(
                "Tensor '{name}' has corrupted offset data: end {end} < start {start}"
            )));
        }
        if end - start > MAX_TENSOR_SIZE {
            return Err(Error::ValidationError(format!(
                "Tensor '{name}' size {} exceeds maximum allowed size of {MAX_TENSOR_SIZE} bytes (potential DoS attack)",
                end - start
            )));
        }
        Ok((start, end))
    }

    /// Produce one [`Bytes`] per requested range, in the same order as `ranges`.
    ///
    /// File-backed ranges are lazy ([`Bytes::from_file`]); in-memory ranges are copied out
    /// of the source buffer.
    fn slice_tensors(&self, ranges: &[(usize, usize)]) -> Result<Vec<Bytes>, Error> {
        match &self.source {
            #[cfg(feature = "std")]
            Source::File(path) => Ok(ranges
                .iter()
                .map(|&(start, end)| {
                    Bytes::from_file(path.clone(), (end - start) as u64, start as u64)
                })
                .collect()),
            Source::Memory(source) => {
                let data: &[u8] = source;
                ranges
                    .iter()
                    .map(|&(start, end)| {
                        let chunk = data.get(start..end).ok_or_else(|| {
                            Error::ValidationError(format!(
                                "Tensor data range {start}..{end} is out of bounds (buffer is {} bytes)",
                                data.len()
                            ))
                        })?;
                        Ok(Bytes::from_bytes_vec(chunk.to_vec()))
                    })
                    .collect()
            }
        }
    }
}

/// Parse and validate a header from a buffer that starts with it.
fn read_header(buf: &[u8]) -> Result<Header, Error> {
    if buf.len() < HEADER_SIZE {
        return Err(Error::InvalidHeader);
    }
    let header = Header::from_bytes(&buf[..HEADER_SIZE])?;
    if header.version > FORMAT_VERSION {
        return Err(Error::InvalidVersion);
    }
    if header.metadata_size > MAX_METADATA_SIZE {
        return Err(Error::ValidationError(format!(
            "Metadata size {} exceeds maximum allowed size of {MAX_METADATA_SIZE} bytes (potential DoS attack)",
            header.metadata_size
        )));
    }
    Ok(header)
}

/// Deserialize the CBOR metadata and validate the tensor count.
fn parse_metadata(bytes: &[u8]) -> Result<Metadata, Error> {
    let metadata: Metadata =
        ciborium::de::from_reader_with_recursion_limit(bytes, MAX_CBOR_RECURSION_DEPTH)
            .map_err(|e| Error::MetadataDeserializationError(e.to_string()))?;
    if metadata.tensors.len() > MAX_TENSOR_COUNT {
        return Err(Error::ValidationError(format!(
            "File contains {} tensors, exceeding maximum of {MAX_TENSOR_COUNT} (potential DoS attack)",
            metadata.tensors.len()
        )));
    }
    Ok(metadata)
}

/// Ensure the available bytes can hold every tensor the metadata claims.
fn validate_total_size(
    metadata: &Metadata,
    metadata_end: usize,
    available: usize,
) -> Result<(), Error> {
    if metadata.tensors.is_empty() {
        return Ok(());
    }
    let max_offset = metadata
        .tensors
        .values()
        .map(|t| t.data_offsets.1)
        .max()
        .unwrap_or(0);
    let max_offset: usize = max_offset.try_into().map_err(|_| {
        Error::ValidationError(format!("Data offset {max_offset} exceeds platform maximum"))
    })?;
    let min_size = metadata_end
        .checked_add(max_offset)
        .ok_or_else(|| Error::ValidationError("File size calculation overflow".into()))?;
    if available < min_size {
        return Err(Error::ValidationError(format!(
            "File truncated: expected at least {min_size} bytes, got {available} bytes"
        )));
    }
    Ok(())
}

/// Build a [`Tensor`] entry from a descriptor + its data bytes.
fn make_tensor(name: &str, descriptor: &TensorDescriptor, bytes: Bytes) -> Result<Tensor, Error> {
    let shape = descriptor
        .shape
        .iter()
        .map(|&s| {
            s.try_into().map_err(|_| {
                Error::ValidationError(format!(
                    "Tensor '{name}' has corrupted shape data: dimension {s} exceeds platform maximum"
                ))
            })
        })
        .collect::<Result<Vec<usize>, Error>>()?;

    Ok(Tensor::new(
        name.to_string(),
        descriptor.dtype,
        Shape::from(shape),
        descriptor.param_id,
        bytes,
    ))
}

#[cfg(feature = "std")]
fn io_err(e: std::io::Error) -> Error {
    Error::IoError(e.to_string())
}

// Verifies the on-disk layout invariant (256-byte tensor alignment), which needs access to the
// internal `TensorDescriptor` offsets. Public round-trip/error tests live in `tests/`.
#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{TENSOR_ALIGNMENT, Tensor, Writer};
    use burn_std::DType;

    fn tensor(name: &str, elems: usize) -> Tensor {
        Tensor::new(
            name.to_string(),
            DType::F32,
            alloc::vec![elems],
            None,
            Bytes::from_bytes_vec(alloc::vec![0u8; elems * 4]),
        )
    }

    #[test]
    fn tensor_offsets_are_256_aligned() {
        // Odd sizes force the writer to insert padding between tensors.
        let packed = Writer::new(vec![tensor("a", 3), tensor("b", 1), tensor("c", 2)])
            .to_bytes()
            .unwrap();
        let reader = Reader::from_bytes(packed).unwrap();

        for (name, descriptor) in &reader.metadata.tensors {
            assert_eq!(
                descriptor.data_offsets.0 % TENSOR_ALIGNMENT,
                0,
                "tensor '{name}' start offset is not 256-aligned"
            );
        }
    }
}
