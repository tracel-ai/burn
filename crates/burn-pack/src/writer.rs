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

/// What [`Writer::build_descriptors`] produces: descriptors keyed by name for the metadata
/// blob, each tensor's [`Placement`] in write order, and the total size of the data section.
type Descriptors = (BTreeMap<String, TensorDescriptor>, Vec<Placement>, usize);

/// Writer for creating Burnpack files.
///
/// Takes the tensors to write, each carrying bytes that are either already resident or
/// produced on demand ([`Tensor::deferred`]). Deferred tensors are drawn one at a time
/// during the write, so a model need not fit in memory to be saved.
pub struct Writer {
    /// Tensors to write
    pub(crate) tensors: Vec<Tensor>,
    /// Metadata key-value pairs
    pub(crate) metadata: BTreeMap<String, String>,
    /// Typed scalars keyed by name
    pub(crate) scalars: BTreeMap<String, Scalar>,
    /// Automatically append the canonical extension to extensionless file paths.
    #[cfg(feature = "std")]
    auto_extension: bool,
}

impl Writer {
    /// Create a new writer
    pub fn new(tensors: Vec<Tensor>) -> Self {
        Self {
            tensors,
            metadata: BTreeMap::new(),
            scalars: BTreeMap::new(),
            #[cfg(feature = "std")]
            auto_extension: true,
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

    /// Enable or disable automatic extension appending for file writes.
    ///
    /// When enabled (the default), [`write_to_file`](Self::write_to_file) and
    /// [`write_to_file_atomic`](Self::write_to_file_atomic) append the canonical
    /// [`crate::EXTENSION`] when the requested path has no extension. When disabled,
    /// both methods use the requested path exactly as provided.
    #[cfg(feature = "std")]
    pub fn auto_extension(mut self, enable: bool) -> Self {
        self.auto_extension = enable;
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
    /// On failure the buffer's contents are unspecified: a deferred [`Tensor`] produces its bytes
    /// during the write, so an entry that fails partway leaves everything before it already
    /// copied in. Callers reusing a buffer across writes cannot treat an error as "nothing
    /// happened". [`write_to_file_atomic`](Self::write_to_file_atomic) has no such caveat;
    /// [`write_to_file`](Self::write_to_file) has the same one, on the destination itself.
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

    /// Write directly to a file, replacing its contents in place.
    ///
    /// By default, the canonical [`crate::EXTENSION`] (`.bpk`) is appended when `path` has no
    /// extension. Use [`auto_extension(false)`](Self::auto_extension) to preserve the path.
    ///
    /// The file is truncated as soon as writing starts, so a failure partway through leaves it
    /// truncated. That only matters when a tensor's bytes can fail to materialize, which for
    /// resident tensors they cannot: the write fails only if the disk does. Callers holding
    /// [`deferred`](Tensor::deferred) tensors, whose providers run mid-write, want
    /// [`write_to_file_atomic`](Self::write_to_file_atomic) instead.
    #[cfg(feature = "std")]
    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<(), Error> {
        let path = self.resolve_path(path.as_ref());
        let layout = self.plan()?;

        let file = File::create(&path)
            .map_err(|e| Error::IoError(format!("cannot create '{}': {e}", path.display())))?;
        let mut sink = FileSink { file, path };

        self.write_container(&layout, &mut sink)
    }

    /// Write to a file without ever leaving a partial one at `path`.
    ///
    /// By default, the canonical [`crate::EXTENSION`] (`.bpk`) is appended when `path` has no
    /// extension. Use [`auto_extension(false)`](Self::auto_extension) to preserve the path.
    ///
    /// The container is built in a scratch sibling of `path` and renamed into place only once
    /// every byte is on disk, so `path` either ends up holding a complete container or is left
    /// exactly as it was. This is what [`deferred`](Tensor::deferred) tensors need: their bytes
    /// are produced during the write, so a provider that fails partway through (or hands back a
    /// different length than it declared) is an ordinary error, and it must not leave a
    /// truncated file where a valid one used to be.
    ///
    /// That much holds everywhere, for failure at the process level: a returned error, a panic,
    /// the process being killed. The rename is a single call, so it either took effect or it
    /// did not, and neither outcome is a partial file.
    ///
    /// Power loss is narrower, and Unix-only. There the data is synced before the rename and
    /// the parent directory after it, so a crash mid-save leaves the old container and a crash
    /// after `Ok` leaves the new one - never a mixture, and never a lost save. Elsewhere both
    /// halves are missing: the directory sync is unavailable, and the replace carries no
    /// documented atomicity guarantee (Windows `MoveFileEx` is not specified as a single
    /// metadata transaction when it replaces an existing file). After power loss the
    /// destination may hold the old container or the new one, and the finished scratch file may
    /// still be beside it.
    ///
    /// Building alongside the destination has four consequences, which is why
    /// [`write_to_file`](Self::write_to_file) does not do it:
    ///
    /// - The data is fsynced before the rename, so the call does not return until the bytes are
    ///   durable rather than merely handed to the page cache.
    /// - Overwriting needs room for a second copy. Re-saving a model over itself transiently
    ///   occupies twice its size, since the old file keeps its blocks until the rename.
    /// - The destination is replaced rather than truncated, so its permissions, ownership and
    ///   hard links do not carry over; the new file gets the process umask. A symlink at `path`
    ///   is replaced by a regular file rather than followed.
    /// - A hard kill (SIGKILL, OOM) skips the cleanup and strands the scratch file. Scratch
    ///   names are `<file_name>.<pid>-<n>.tmp` siblings of the resolved path (after any
    ///   extension is appended), so leftovers are identifiable and safe to delete once no
    ///   writer is running.
    #[cfg(feature = "std")]
    pub fn write_to_file_atomic<P: AsRef<Path>>(self, path: P) -> Result<(), Error> {
        let path = self.resolve_path(path.as_ref());
        let layout = self.plan()?;
        let (scratch, mut sink) = ScratchFile::create(&path)?;

        self.write_container(&layout, &mut sink)?;

        scratch.persist(sink, &path)
    }

    /// Apply the configured extension policy to a requested path.
    #[cfg(feature = "std")]
    fn resolve_path(&self, path: &Path) -> std::path::PathBuf {
        if self.auto_extension && path.extension().is_none() {
            path.with_extension(crate::EXTENSION)
        } else {
            path.to_path_buf()
        }
    }

    /// Build the complete on-disk layout: header, serialized metadata, and the
    /// position and size of the (aligned) tensor data section.
    fn plan(&self) -> Result<Layout, Error> {
        let (tensors, placements, data_size) = self.build_descriptors()?;
        let metadata = Metadata {
            tensors,
            metadata: self.metadata.clone(),
            scalars: self.scalars.clone(),
        };

        let mut metadata_bytes = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut metadata_bytes)
            .map_err(|e| Error::MetadataSerializationError(e.to_string()))?;

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

    /// Build tensor descriptors, assigning each tensor an aligned offset within
    /// the data section so that absolute file positions are mmap-friendly.
    ///
    /// Returns the descriptors keyed by name (for the metadata blob), the [`Placement`] of
    /// each tensor in `self.tensors` order (for the write pass), and the total data-section
    /// size (the running offset after the last tensor). Offsets only grow, so this is also
    /// the highest descriptor end offset.
    fn build_descriptors(&self) -> Result<Descriptors, Error> {
        let mut tensors = BTreeMap::new();
        let mut placements = Vec::with_capacity(self.tensors.len());
        let mut current_offset = 0u64;

        for tensor in &self.tensors {
            let name = &tensor.name;
            let dtype = tensor.dtype;
            let shape = &tensor.shape;
            let data_len = tensor.byte_len() as u64;

            Self::check_byte_len(name, dtype, shape, data_len)?;

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
                        param_id: tensor.param_id,
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
                name: name.clone(),
                offset: aligned_start as usize,
            });
            current_offset = end;
        }

        Ok((tensors, placements, current_offset as usize))
    }

    /// Reject a `byte_len` that cannot describe the tensor the entry declares.
    ///
    /// A reader sizes a tensor's [`Bytes`] from the descriptor's shape and
    /// dtype, so a length that disagrees with them produces a container that reads back
    /// wrong rather than one that fails to write. Catching it during planning costs nothing
    /// and keeps the mistake from reaching a file at all.
    ///
    /// Quantized data is the deliberate exception: values are bit-packed and the scales are
    /// appended inline, so its length is not `num_elements * dtype.size()`. That layout
    /// contract belongs to the quantization layer, and burn-pack deliberately does not
    /// reimplement it - which is why [`Tensor::deferred`] takes a byte length the
    /// producer supplies rather than something derived here.
    fn check_byte_len(name: &str, dtype: DType, shape: &Shape, data_len: u64) -> Result<(), Error> {
        if matches!(dtype, DType::QFloat(_)) {
            return Ok(());
        }

        let expected = shape.num_elements() as u64 * dtype.size() as u64;
        if data_len != expected {
            return Err(Error::ValidationError(format!(
                "tensor '{}' declares {} bytes but its shape {} and dtype {:?} need {}",
                name, data_len, shape, dtype, expected
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
    /// two pairs each tensor with its own offset by construction, with no lookup to get out
    /// of step.
    fn write_tensors(self, placements: &[Placement], sink: &mut impl Sink) -> Result<(), Error> {
        // The zip below would silently drop tensors if the two ever diverged in length, and
        // that is a corrupt container; one integer comparison per write buys a loud abort
        // (which the scratch-file guard turns into a clean one) in release builds too.
        assert_eq!(placements.len(), self.tensors.len());

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
    /// from [`Tensor::byte_len`] long before these bytes existed, so a provider that
    /// reports one size and produces another would misplace every tensor after it.
    fn materialize(tensor: Tensor, placement: &Placement) -> Result<Bytes, Error> {
        // Name the tensor on the way out. `into_bytes` runs mid-write, so its failures arrive
        // interleaved with the writer's own disk errors; without this, a device readback that
        // fails on one tensor of eight hundred is indistinguishable from a full disk.
        tensor
            .into_bytes()
            .map_err(|e| e.in_tensor(&placement.name))
    }
}

/// Where one tensor's bytes belong in the data section, computed during planning and
/// carried forward so the write pass needs nothing from the tensor but its bytes.
struct Placement {
    /// The tensor's name, for error messages after the tensor has been consumed.
    name: String,
    /// Aligned start, relative to the beginning of the data section.
    offset: usize,
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
/// via [`Writer::plan`] and shared by `size`, `write_into`, `to_bytes`, `write_to_file` and
/// `write_to_file_atomic`.
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
/// Gives [`Writer::write_to_file_atomic`] its all-or-nothing behaviour: the container is built
/// here and only takes the destination's name once it is complete. Any earlier return
/// drops the guard, which removes the partial file.
///
/// The scratch path is a sibling of the destination rather than a system temp file so that
/// [`persist`](Self::persist) stays on one filesystem: never `EXDEV`, and atomic on POSIX.
/// (Windows documents no atomicity guarantee when the rename replaces an existing file.)
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
    ///
    /// The file is opened with `create_new`, which refuses a path that already exists and
    /// does not follow a symlink planted there - a plain `File::create` would truncate
    /// whatever such a link points at. The scratch name is predictable (pid + counter), so a
    /// stale leftover from a killed run, pid reuse, or a deliberately pre-created link all
    /// surface as `AlreadyExists`, answered by moving on to the next counter value.
    fn create(destination: &Path) -> Result<(Self, FileSink), Error> {
        loop {
            let path = Self::path_beside(destination)?;
            match File::create_new(&path) {
                Ok(file) => {
                    let sink = FileSink {
                        file,
                        path: path.clone(),
                    };
                    return Ok((
                        Self {
                            path,
                            persisted: false,
                        },
                        sink,
                    ));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(Error::IoError(format!(
                        "cannot create '{}': {e}",
                        path.display()
                    )));
                }
            }
        }
    }

    /// Pick a scratch path alongside `destination`.
    ///
    /// The name carries the process id and a counter so concurrent writers for the same
    /// destination pick distinct names; [`create`](Self::create)'s exclusive open is what
    /// makes the rare collision harmless rather than this scheme's uniqueness.
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

    /// Flush the completed container to disk and move it onto `destination`, replacing any
    /// existing file there.
    ///
    /// Taking the sink is what enforces the ordering the all-or-nothing guarantee rests on:
    /// persisting without first surrendering the file handle is unrepresentable, so the
    /// deferred-write-error check in [`FileSink::finish`] cannot be skipped and the handle is
    /// closed before the rename.
    ///
    /// On Unix the parent directory is synced after the rename, making the rename itself
    /// durable: once this returns `Ok`, power loss cannot revert the destination to the old
    /// container. (Windows has no portable directory sync; NTFS journals metadata on its own
    /// schedule.) If that final sync fails, the error says so explicitly - the new container
    /// is at the destination and intact, only its durability is unconfirmed - because the
    /// generic rename error below would wrongly imply the save did not happen.
    ///
    /// On rename failure the finished container is deleted by the guard, so the error names
    /// it: the bytes were written and then discarded, which is worth saying out loud.
    fn persist(mut self, sink: FileSink, destination: &Path) -> Result<(), Error> {
        sink.finish()?;

        std::fs::rename(&self.path, destination).map_err(|e| {
            Error::IoError(format!(
                "cannot move the completed container '{}' onto '{}': {e}",
                self.path.display(),
                destination.display()
            ))
        })?;
        self.persisted = true;

        #[cfg(unix)]
        {
            // An empty parent means a bare relative file name; the directory is the cwd.
            let parent = match destination.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent,
                _ => Path::new("."),
            };
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|e| {
                    Error::IoError(format!(
                        "'{}' was saved, but syncing its directory failed, so the rename may \
                         not survive power loss: {e}",
                        destination.display()
                    ))
                })?;
        }

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
    /// while `write_to_file_atomic` returned `Ok`.
    ///
    /// It also orders durability: the data is on disk before the rename happens, so a crash
    /// cannot leave the destination pointing at data that never reached the platter.
    /// (Durability of the rename itself is [`ScratchFile::persist`]'s job.)
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
