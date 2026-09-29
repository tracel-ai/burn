use super::base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAX_CBOR_RECURSION_DEPTH, MAX_METADATA_SIZE,
    MAX_TENSOR_COUNT, MAX_TENSOR_SIZE, Metadata, TensorDescriptor, aligned_data_section_start,
    validate_tensor_byte_len,
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
use std::path::Path;

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
    /// Loading is lazy: only the header and metadata are parsed here. The buffer is kept as-is
    /// (no copy, no share) and is only turned into zero-copy [`Bytes::view`] windows when you
    /// consume the reader with [`into_tensors`](Self::into_tensors). For large models, prefer
    /// [`from_file`](Self::from_file), which keeps tensor data file-backed and lazy.
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
    /// Only the header and metadata are read up front; the whole file is wrapped in a single
    /// lazy [`Bytes::from_file`] source, and each tensor's data is a [`Bytes::view`] window into
    /// it, read from disk only when accessed. This integrates with the Burn ecosystem's file
    /// allocation (pinned-memory staging) for fast file-to-GPU transfers.
    ///
    /// If `path` has no extension and does not exist as given, the canonical
    /// [`crate::EXTENSION`] (`.bpk`) is appended.
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path.as_ref();
        let path = if path.extension().is_none() && !path.exists() {
            path.with_extension(crate::EXTENSION)
        } else {
            path.to_path_buf()
        };

        Self::from_resolved_file(&path)
    }

    /// Load a pack from a file using the exact path provided.
    ///
    /// Unlike [`from_file`](Self::from_file), this never appends [`crate::EXTENSION`] when the
    /// requested path has no extension and does not probe for a fallback path. It is useful when
    /// a higher-level caller has already applied its own path policy.
    #[cfg(feature = "std")]
    pub fn from_file_exact<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        Self::from_resolved_file(path.as_ref())
    }

    #[cfg(feature = "std")]
    fn from_resolved_file(path: &Path) -> Result<Self, Error> {
        let mut file = File::open(path)
            .map_err(|e| Error::IoError(format!("cannot open '{}': {e}", path.display())))?;

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

        let source = Source::File(Bytes::from_file(path, file_size, 0));
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
        let data_offset = aligned_data_section_start(header.metadata_size as usize);
        validate_total_size(&metadata, data_offset, available)?;

        Ok(Self {
            metadata,
            source,
            data_offset,
        })
    }

    /// Consume the reader, returning all tensors in sorted (alphabetical) name order.
    ///
    /// Each tensor's bytes are a zero-copy [`Bytes::view`] window into the source — no per-tensor
    /// copy. For an in-memory source the buffer is [shared](Bytes::shared) once here (a cheap
    /// `Arc` move, never a data copy) to make those views available; for a file source the windows
    /// are file-backed and read lazily on access. Consuming `self` lets us hand the source's
    /// ownership to the views directly, so loading never reads or copies tensor data eagerly.
    pub fn into_tensors(self) -> Result<Vec<Tensor>, Error> {
        let Reader {
            metadata,
            source,
            data_offset,
        } = self;

        // Make the source view-capable: a plain in-memory buffer has no zero-copy window until
        // it's shared behind an `Arc`, whereas a file-backed source already windows lazily (and
        // must NOT be shared, or every view would materialize the whole file).
        let source = match source {
            Source::Memory(bytes) => bytes.shared(),
            #[cfg(feature = "std")]
            Source::File(bytes) => bytes,
        };

        let mut tensors = Vec::with_capacity(metadata.tensors.len());
        for (name, descriptor) in &metadata.tensors {
            let (start, end) = tensor_range(data_offset, name, descriptor)?;
            let bytes = source.view(start, end).map_err(|_| {
                Error::ValidationError(format!(
                    "Tensor '{name}' data range {start}..{end} could not be viewed (source is {} bytes)",
                    source.len()
                ))
            })?;
            tensors.push(make_tensor(name, descriptor, bytes)?);
        }
        Ok(tensors)
    }

    /// The user-supplied key/value metadata stored alongside the tensors.
    ///
    /// For per-tensor info (dtype/shape/param id), use [`into_tensors`](Self::into_tensors) — for a
    /// file-backed reader that does not read any tensor data until a tensor's bytes are accessed.
    pub fn metadata(&self) -> &alloc::collections::BTreeMap<String, String> {
        &self.metadata.metadata
    }

    /// The typed scalars stored alongside the tensors.
    ///
    /// Empty for files written before scalar support.
    pub fn scalars(&self) -> &alloc::collections::BTreeMap<String, crate::Scalar> {
        &self.metadata.scalars
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
        let (start, end) = tensor_range(self.data_offset, name, descriptor)?;

        match &self.source {
            #[cfg(feature = "std")]
            Source::File(bytes) => {
                // A file-backed view reads just this tensor's range from disk.
                let view = bytes.view(start, end).map_err(|_| {
                    Error::ValidationError(format!(
                        "Tensor '{name}' data range {start}..{end} could not be viewed"
                    ))
                })?;
                let slice: &[u8] = &view;
                Ok(slice.to_vec())
            }
            Source::Memory(bytes) => Ok(memory_chunk(bytes, start, end)?.to_vec()),
        }
    }
}

/// Compute and validate the absolute `[start, end)` byte range of a tensor.
fn tensor_range(
    data_offset: usize,
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

    let start = data_offset
        .checked_add(to_usize(descriptor.data_offsets.0)?)
        .ok_or_else(overflow)?;
    let end = data_offset
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
    data_offset: usize,
    available: usize,
) -> Result<(), Error> {
    let mut min_size = None;
    for (name, descriptor) in &metadata.tensors {
        let (start, end) = tensor_range(data_offset, name, descriptor)?;
        validate_tensor_byte_len(
            name,
            descriptor.dtype,
            &descriptor.shape,
            (end - start) as u64,
        )?;
        min_size = Some(min_size.map_or(end, |current: usize| current.max(end)));
    }

    let Some(min_size) = min_size else {
        return Ok(());
    };
    if available < min_size {
        return Err(Error::ValidationError(format!(
            "File truncated: expected at least {min_size} bytes, got {available} bytes"
        )));
    }
    Ok(())
}

/// Borrow a tensor's `[start, end)` range out of an in-memory buffer, bounds-checked.
fn memory_chunk(source: &Bytes, start: usize, end: usize) -> Result<&[u8], Error> {
    let data: &[u8] = source;
    data.get(start..end).ok_or_else(|| {
        Error::ValidationError(format!(
            "Tensor data range {start}..{end} is out of bounds (buffer is {} bytes)",
            data.len()
        ))
    })
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

/// Where a [`Reader`] gets its tensor data from.
///
/// Both variants hold a single [`Bytes`] spanning the whole container, and tensors are carved
/// out of it with zero-copy [`Bytes::view`] windows.
enum Source {
    /// The whole pack lives in memory.
    Memory(Bytes),
    /// The pack lives in a file; tensor data is read lazily via file-backed [`Bytes::view`].
    #[cfg(feature = "std")]
    File(Bytes),
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
    use alloc::collections::BTreeMap;
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

    fn pack_with_descriptor(shape: Vec<u64>, data_len: u64) -> Bytes {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "weight".to_string(),
            TensorDescriptor {
                dtype: DType::F32,
                shape,
                data_offsets: (0, data_len),
                param_id: None,
            },
        );
        let metadata = Metadata {
            tensors,
            metadata: BTreeMap::new(),
            scalars: BTreeMap::new(),
        };
        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes).unwrap();
        let header = Header::new(metadata_bytes.len() as u32);
        let data_offset = aligned_data_section_start(metadata_bytes.len());

        let mut bytes = header.into_bytes().to_vec();
        bytes.extend(metadata_bytes);
        bytes.resize(data_offset + data_len as usize, 0);
        Bytes::from_bytes_vec(bytes)
    }

    fn assert_byte_length_error(bytes: Bytes, expected: &str) {
        match Reader::from_bytes(bytes) {
            Err(Error::ValidationError(message)) => assert!(
                message.contains(expected),
                "expected error containing {expected:?}, got {message:?}"
            ),
            _ => panic!("expected a byte-length validation error"),
        }
    }

    #[test]
    fn tensor_offsets_are_256_aligned() {
        // Odd sizes force the writer to insert padding between tensors.
        let packed = Writer::new(vec![tensor("a", 3), tensor("b", 1), tensor("c", 2)])
            .into_bytes()
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

    #[test]
    fn rejects_tensor_data_shorter_than_declared_shape() {
        assert_byte_length_error(pack_with_descriptor(vec![2, 3], 16), "need 24");
    }

    #[test]
    fn rejects_tensor_data_longer_than_declared_shape() {
        assert_byte_length_error(pack_with_descriptor(vec![2, 3], 32), "need 24");
    }

    #[test]
    fn rejects_tensor_byte_length_overflow() {
        assert_byte_length_error(
            pack_with_descriptor(vec![u64::MAX, 2], 0),
            "calculation overflows u64",
        );
    }
}
