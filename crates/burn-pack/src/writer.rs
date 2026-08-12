use super::base::{
    Error, FORMAT_VERSION, HEADER_SIZE, Header, MAGIC_NUMBER, Metadata, Scalar, TENSOR_ALIGNMENT,
    TensorDescriptor, aligned_data_section_start,
};
use super::tensor::{Tensor, TensorEntry};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Bytes, DType, Shape};

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

/// What planning produces: descriptors keyed by name for the metadata blob, each tensor's
/// [`Placement`] in write order, and the total size of the data section.
type Descriptors = (BTreeMap<String, TensorDescriptor>, Vec<Placement>, usize);

/// Writer for creating Burnpack files.
///
/// Generic over the [`TensorEntry`] implementation supplying the tensors; the default,
/// [`Tensor`], carries bytes that are already resident. See [`TensorEntry`] for how to
/// supply tensors that materialize one at a time instead.
pub struct Writer<T: TensorEntry = Tensor> {
    /// Tensors to write
    pub(crate) tensors: Vec<T>,
    /// Metadata key-value pairs
    pub(crate) metadata: BTreeMap<String, String>,
    /// Typed scalars keyed by name
    pub(crate) scalars: BTreeMap<String, Scalar>,
}

impl<T: TensorEntry> Writer<T> {
    /// Create a new writer
    pub fn new(tensors: Vec<T>) -> Self {
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
    /// On failure the buffer's contents are unspecified: a [`TensorEntry`] produces its bytes
    /// during the write, so an entry that fails partway leaves everything before it already
    /// copied in. Callers reusing a buffer across writes cannot treat an error as "nothing
    /// happened". [`write_to_file`](Self::write_to_file) has no such caveat.
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

    /// Write directly to a file (more memory efficient for large models).
    ///
    /// If `path` has no extension, the canonical [`crate::EXTENSION`] (`.bpk`) is appended.
    ///
    /// The container is written to a scratch sibling of `path` and renamed into place only
    /// once every byte is on disk, so `path` either ends up holding a complete container or
    /// is left exactly as it was. This matters for lazy [`TensorEntry`] implementations,
    /// whose bytes are produced during the write: a provider that fails partway through (or
    /// hands back a different length than it declared) is an ordinary error, and it must not
    /// leave a truncated file where a valid one used to be.
    ///
    /// The guarantee covers process-level failure: a returned error, a panic, or the process
    /// dying. It does not extend to power loss or a kernel panic, where the rename may reach
    /// the disk before the data blocks do.
    ///
    /// Building alongside the destination has four consequences:
    ///
    /// - Overwriting needs room for a second copy. Re-saving a model over itself transiently
    ///   occupies twice its size, since the old file keeps its blocks until the rename.
    /// - A hard kill (SIGKILL, OOM) skips the cleanup and strands the scratch file. Scratch
    ///   names are `<file_name>.<pid>-<n>.tmp` siblings of the resolved path (after any
    ///   extension is appended), so leftovers are identifiable and safe to delete once no
    ///   writer is running.
    /// - The destination is replaced rather than truncated, so its permissions, ownership and
    ///   hard links do not carry over; the new file gets the process umask.
    /// - A symlink at `path` is replaced by a regular file rather than followed. Saving over
    ///   a symlink that points at bulk storage stops updating the target.
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<(), Error> {
        let path = path.as_ref();
        let path = if path.extension().is_none() {
            path.with_extension(crate::EXTENSION)
        } else {
            path.to_path_buf()
        };

        let layout = self.plan()?;
        let (scratch, mut sink) = ScratchFile::create(&path)?;

        self.write_container(&layout, &mut sink)?;
        sink.finish()?;

        scratch.persist(&path)
    }

    /// Build the complete on-disk layout: header, serialized metadata, and the
    /// position and size of the (aligned) tensor data section.
    fn plan(&self) -> Result<Layout, Error> {
        let (metadata_bytes, placements, data_size) = self.build_metadata()?;

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
            metadata_bytes,
            placements,
            header,
            data_section_start,
            data_size,
        })
    }

    /// Serialize the metadata structure (tensor descriptors + key-value pairs) to CBOR.
    ///
    /// Also returns the per-tensor placements and the size of the tensor data section, both
    /// computed while assigning offsets.
    fn build_metadata(&self) -> Result<(Vec<u8>, Vec<Placement>, usize), Error> {
        let (tensors, placements, data_size) = self.build_descriptors()?;
        let metadata = Metadata {
            tensors,
            metadata: self.metadata.clone(),
            scalars: self.scalars.clone(),
        };

        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| Error::MetadataSerializationError(e.to_string()))?;

        Ok((metadata_bytes, placements, data_size))
    }

    /// Build tensor descriptors, assigning each tensor an aligned offset within
    /// the data section so that absolute file positions are mmap-friendly.
    ///
    /// Returns the descriptors keyed by name (for the metadata blob), the [`Placement`] of
    /// each tensor in `self.tensors` order (for the write pass), and the total data-section
    /// size — the running offset after the last tensor. Offsets only grow, so this is also
    /// the highest descriptor end offset.
    fn build_descriptors(&self) -> Result<Descriptors, Error> {
        let mut tensors = BTreeMap::new();
        let mut placements = Vec::with_capacity(self.tensors.len());
        let mut current_offset = 0u64;

        for tensor in &self.tensors {
            // Read every accessor exactly once. The write pass works from the `Placement`
            // recorded here rather than calling back into the entry, so an implementation
            // whose accessors are not stable still cannot desynchronize the two passes.
            let name = tensor.name().into_owned();
            let dtype = tensor.dtype();
            let shape = tensor.shape();
            let data_len = tensor.byte_len() as u64;

            Self::check_byte_len(&name, dtype, shape, data_len)?;

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
                    name.clone(),
                    TensorDescriptor {
                        dtype,
                        shape: shape.iter().map(|&s| s as u64).collect(),
                        data_offsets: (aligned_start, end),
                        param_id: tensor.param_id(),
                    },
                )
                .is_some()
            {
                return Err(Error::ValidationError(format!(
                    "Duplicate tensor name '{}'",
                    name
                )));
            }

            placements.push(Placement {
                name,
                offset: aligned_start as usize,
                len: data_len as usize,
            });
            current_offset = end;
        }

        Ok((tensors, placements, current_offset as usize))
    }

    /// Reject a `byte_len` that cannot describe the tensor the entry declares.
    ///
    /// A reader sizes its [`TensorData`](burn_std::Bytes) from the descriptor's shape and
    /// dtype, so a length that disagrees with them produces a container that reads back
    /// wrong rather than one that fails to write. Catching it during planning costs nothing
    /// and keeps the mistake from reaching a file at all.
    ///
    /// Quantized data is the deliberate exception: values are bit-packed and the scales are
    /// appended inline, so its length is not a product of shape and dtype and only the
    /// producer can compute it. That exception is why [`TensorEntry::byte_len`] exists as a
    /// method rather than being derived here.
    fn check_byte_len(name: &str, dtype: DType, shape: &Shape, data_len: u64) -> Result<(), Error> {
        if matches!(dtype, DType::QFloat(_)) {
            return Ok(());
        }

        let expected = shape.iter().product::<usize>() as u64 * dtype.size() as u64;
        if data_len != expected {
            return Err(Error::ValidationError(format!(
                "tensor '{}' declares {} bytes but its shape {:?} and dtype {:?} need {}",
                name,
                data_len,
                shape.to_vec(),
                dtype,
                expected
            )));
        }

        Ok(())
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

        self.write_tensors(&layout.placements, sink)
    }

    /// Write each tensor's data into `sink`, inserting alignment padding between
    /// tensors so every tensor lands at its planned offset.
    ///
    /// `placements` was built by walking `self.tensors` in this same order, so zipping the
    /// two pairs each entry with its own offset by construction. Offsets are never recovered
    /// from the entry a second time, which is what keeps the plan and the write in step.
    fn write_tensors(self, placements: &[Placement], sink: &mut impl Sink) -> Result<(), Error> {
        // Position within the data section (relative to its aligned start).
        let mut data_offset = 0usize;

        for (tensor, placement) in self.tensors.into_iter().zip(placements) {
            if placement.offset > data_offset {
                sink.pad(placement.offset - data_offset)?;
                data_offset = placement.offset;
            }

            let data = Self::materialize(tensor, placement)?;
            write_tensor_data(&data, sink)?;
            data_offset += data.len();
        }

        Ok(())
    }

    /// Materialize one tensor's bytes and check they fill exactly the space reserved for it.
    ///
    /// The length check is what keeps a lazy entry honest: the offset table was committed
    /// from [`TensorEntry::byte_len`] long before these bytes existed, so a provider that
    /// reports one size and produces another would misplace every tensor after it.
    fn materialize(tensor: T, placement: &Placement) -> Result<Bytes, Error> {
        // Name the tensor on the way out. `into_bytes` runs mid-write, so its failures arrive
        // interleaved with the writer's own disk errors; without this, a device readback that
        // fails on one tensor of eight hundred is indistinguishable from a full disk.
        let bytes = tensor
            .into_bytes()
            .map_err(|e| e.in_tensor(&placement.name))?;

        if bytes.len() != placement.len {
            return Err(Error::TensorBytesSizeMismatch(format!(
                "tensor '{}' has inconsistent length (expected {}, got {})",
                placement.name,
                placement.len,
                bytes.len()
            )));
        }

        Ok(bytes)
    }
}

/// Where one tensor's bytes belong in the data section, recorded during planning.
///
/// Planning and writing are separate passes over the same `Vec<T>`, and only the first may
/// touch the entries' accessors. Carrying the result forward means the second pass needs
/// nothing from the entry but its bytes.
struct Placement {
    /// The tensor's name, for error messages after the entry has been consumed.
    name: String,
    /// Aligned start, relative to the beginning of the data section.
    offset: usize,
    /// Bytes reserved, from [`TensorEntry::byte_len`].
    len: usize,
}

/// Stream a single tensor's bytes into `sink`, materializing at most
/// [`WRITE_CHUNK_SIZE`] bytes at a time.
///
/// When the backing supports zero-copy windows (device-resident
/// [lazy](burn_std::Bytes) device readback, file, or shared buffers), each
/// chunk is taken as a [`Bytes::view`] and read just-in-time, then dropped
/// before the next one. A large device tensor is therefore copied to host in
/// bounded pieces rather than through one big (pinned) staging buffer, so the
/// whole tensor never has to be resident at once.
///
/// Backings without a zero-copy window (e.g. a plain heap `Vec`) are already
/// host-resident, so [`Bytes::view`] reports it can't window them and the
/// remaining bytes are written in a single pass.
///
/// Free-standing rather than a method on [`Writer`]: it does not depend on the entry type,
/// so this way it is compiled once per [`Sink`] instead of once per (entry type, sink) pair.
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

/// The computed on-disk layout of a burnpack container.
///
/// Captures everything needed to emit the bytes: the serialized metadata, the
/// header, where the aligned data section begins, and how large it is. Built once
/// via [`Writer::plan`] and shared by `size`, `write_into`, `to_bytes`, and
/// `write_to_file`.
struct Layout {
    metadata_bytes: Vec<u8>,
    /// Where each tensor's bytes go, in `Writer::tensors` order.
    placements: Vec<Placement>,
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

/// A scratch file next to the eventual destination, deleted unless it is persisted.
///
/// Gives [`Writer::write_to_file`] its all-or-nothing behaviour: the container is built
/// here and only takes the destination's name once it is complete. Any earlier return
/// drops the guard, which removes the partial file.
///
/// The scratch path is a sibling of the destination rather than a system temp file so that
/// [`persist`](Self::persist) stays on one filesystem: never `EXDEV`, and atomic on POSIX.
/// (Windows renames via `MoveFileExW`, which Microsoft does not document as atomic when it
/// replaces an existing file.)
#[cfg(feature = "std")]
struct ScratchFile {
    path: std::path::PathBuf,
    persisted: bool,
}

#[cfg(feature = "std")]
impl ScratchFile {
    /// Create a scratch file alongside `destination` and open a sink onto it.
    ///
    /// The guard and its file are handed back together so neither can exist without the
    /// other: no path is reserved that nothing will clean up, and no file is created that no
    /// guard is watching.
    fn create(destination: &Path) -> Result<(Self, FileSink), Error> {
        let path = Self::path_beside(destination)?;
        let file = File::create(&path)
            .map_err(|e| Error::IoError(format!("cannot create '{}': {e}", path.display())))?;

        let sink = FileSink {
            file,
            path: path.clone(),
        };
        Ok((
            Self {
                path,
                persisted: false,
            },
            sink,
        ))
    }

    /// Pick an unused scratch path alongside `destination`.
    ///
    /// The name carries the process id and a counter so concurrent writers, in this process
    /// or another, never pick the same scratch file for the same destination.
    fn path_beside(destination: &Path) -> Result<std::path::PathBuf, Error> {
        use core::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let name = destination.file_name().ok_or_else(|| {
            Error::IoError(format!(
                "cannot write to '{}': not a file path",
                destination.display()
            ))
        })?;

        let mut scratch = name.to_os_string();
        scratch.push(format!(
            ".{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));

        Ok(destination.with_file_name(scratch))
    }

    /// Move the completed container onto `destination`, replacing any existing file.
    ///
    /// On failure the finished container is deleted by the guard, so the error names it: the
    /// bytes were written and then discarded, which is worth saying out loud.
    fn persist(mut self, destination: &Path) -> Result<(), Error> {
        std::fs::rename(&self.path, destination).map_err(|e| {
            Error::IoError(format!(
                "cannot move the completed container '{}' onto '{}': {e}",
                self.path.display(),
                destination.display()
            ))
        })?;
        self.persisted = true;
        Ok(())
    }
}

#[cfg(feature = "std")]
impl Drop for ScratchFile {
    fn drop(&mut self) {
        if !self.persisted {
            // Best effort: the write already failed, and a leftover scratch file is a less
            // useful thing to report than whatever went wrong.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Sink that streams directly to a file.
///
/// Carries the path so its errors can name the file. The writer works on a scratch file
/// whose name the caller never chose, so "No space left on device" with nothing attached
/// would leave them with no idea which file the writer was even touching.
#[cfg(feature = "std")]
struct FileSink {
    file: File,
    path: std::path::PathBuf,
}

#[cfg(feature = "std")]
impl FileSink {
    /// Flush the container to the device and close the handle.
    ///
    /// This is the only place a deferred write error can still be caught. `File::flush` is a
    /// no-op because the handle is unbuffered, and dropping it discards whatever `close`
    /// reports, yet filesystems that allocate lazily (NFS over quota, a failing disk) report
    /// exactly there. Without this, such a write would be renamed over a good container
    /// while `write_to_file` returned `Ok`.
    ///
    /// It also makes the rename durable in the right order on POSIX, so a crash cannot leave
    /// the destination pointing at data that never reached the platter.
    fn finish(self) -> Result<(), Error> {
        self.file.sync_all().map_err(|e| {
            Error::IoError(format!(
                "cannot flush '{}' to disk: {e}",
                self.path.display()
            ))
        })
    }
}

#[cfg(feature = "std")]
impl Sink for FileSink {
    fn pad(&mut self, count: usize) -> Result<(), Error> {
        // Stream zeros without allocating a `count`-sized buffer per call.
        std::io::copy(&mut std::io::repeat(0).take(count as u64), &mut self.file)
            .map(|_| ())
            .map_err(|e| Error::IoError(format!("cannot write to '{}': {e}", self.path.display())))
    }

    fn write(&mut self, data: &[u8]) -> Result<(), Error> {
        self.file
            .write_all(data)
            .map_err(|e| Error::IoError(format!("cannot write to '{}': {e}", self.path.display())))
    }
}
