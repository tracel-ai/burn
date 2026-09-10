//! Where tensor bytes come from once a pickle has been parsed.
//!
//! Every PyTorch container keeps tensor metadata (a pickle) apart from the storages holding
//! the raw bytes (little-endian in every file this reader accepts). A [`StorageSource`] maps
//! a storage key from that pickle to its bytes on demand: the ZIP and legacy sources read
//! the file at that point, the TAR source slices an entry already in memory.

use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zip::ZipArchive;

use super::reader::PytorchError;
use burn_pack::MAX_METADATA_SIZE;

/// Largest `version`, `byteorder` or similar text entry accepted.
const MAX_TEXT_ENTRY_SIZE: u64 = 4096;

/// Storage bytes for one container format.
pub(crate) enum StorageSource {
    Zip(ZipSource),
    Tar(TarSource),
    Legacy(LegacySource),
}

impl fmt::Debug for StorageSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Zip(_) => "StorageSource::Zip",
            Self::Tar(_) => "StorageSource::Tar",
            Self::Legacy(_) => "StorageSource::Legacy",
        })
    }
}

impl StorageSource {
    /// Read at most `max_len` bytes of the storage saved under `key`, starting at `start`.
    ///
    /// A caller knows the window its tensor can touch, and the rest of the storage is never
    /// held: bytes past the window are left unread, and bytes before it are dropped as they
    /// go by, so a ZIP entry that decompresses far beyond its archive size costs its window
    /// rather than its size. A storage shorter than the window yields what it has.
    ///
    /// Returns the window and the number of bytes that preceded it, which is below `start`
    /// only when the storage ends inside them. The two together say how far the storage
    /// reaches, which a short read is diagnosed with.
    pub fn read(&self, key: &str, start: usize, max_len: usize) -> io::Result<(Vec<u8>, usize)> {
        match self {
            Self::Zip(source) => source.read_storage(key, start, max_len),
            Self::Tar(source) => source.read_storage(key, start, max_len),
            Self::Legacy(source) => source.read_storage(key, start, max_len),
        }
    }

    /// Record the size a pickle declares for a storage.
    ///
    /// Only the legacy container needs this: its storages are concatenated without any
    /// index, so the pickle's declarations are the only way to find their boundaries.
    pub fn declare(&self, key: &str, byte_len: usize, element_size: usize) -> io::Result<()> {
        match self {
            Self::Legacy(source) => source.declare(key, byte_len, element_size),
            Self::Zip(_) | Self::Tar(_) => Ok(()),
        }
    }
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn lock_ignoring_poison<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Read exactly `len` bytes without trusting `len` for the up-front allocation.
///
/// A length read from an untrusted file could be anything, so at most `capacity_bound`
/// bytes are reserved ahead of the read (the caller's idea of the largest plausible size,
/// such as the file length); beyond that the buffer grows only with bytes that actually
/// arrive, and a short read is an error.
pub(crate) fn read_exact_len<R: Read>(
    reader: &mut R,
    len: u64,
    capacity_bound: usize,
) -> io::Result<Vec<u8>> {
    let capacity = usize::try_from(len).map_or(capacity_bound, |len| len.min(capacity_bound));
    let mut buffer = Vec::with_capacity(capacity);
    reader.by_ref().take(len).read_to_end(&mut buffer)?;
    if buffer.len() as u64 != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("expected {len} bytes, found {}", buffer.len()),
        ));
    }
    Ok(buffer)
}

/// The modern ZIP container (PyTorch 1.6+).
///
/// `torch.save` writes every entry under a root directory named after the file
/// (`model/data.pkl`, `model/data/0`, `model/version`, ...). Files saved through a file
/// object or an in-memory buffer use `archive/`, and some tools write the entries at the
/// root. The directory holding `data.pkl` is the root for every other entry.
pub(crate) struct ZipSource {
    archive: Mutex<ZipArchive<BufReader<File>>>,
    /// Root directory including its trailing slash, or empty at the archive root.
    root: String,
    /// Size of the archive file; caps the up-front allocation for an entry (a stored entry
    /// cannot be larger, a deflated one grows as it is read).
    file_len: usize,
}

impl ZipSource {
    pub fn open(path: &Path) -> Result<Self, PytorchError> {
        let file = File::open(path)?;
        let file_len = usize::try_from(file.metadata()?.len()).unwrap_or(usize::MAX);
        let archive = ZipArchive::new(BufReader::new(file))?;

        let root = archive
            .file_names()
            .filter_map(|name| name.strip_suffix("data.pkl"))
            .filter(|root| root.is_empty() || root.ends_with('/'))
            .min_by_key(|root| root.len())
            .map(str::to_string)
            .ok_or_else(|| {
                PytorchError::InvalidFormat(
                    "No data.pkl entry found in ZIP archive. Expected a PyTorch 1.6+ checkpoint"
                        .to_string(),
                )
            })?;

        Ok(Self {
            archive: Mutex::new(archive),
            root,
            file_len,
        })
    }

    /// Total uncompressed size of the storage entries.
    pub fn data_size(&self) -> io::Result<u64> {
        let data_prefix = format!("{}data/", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let indices: Vec<usize> = archive
            .file_names()
            .filter(|name| name.starts_with(&data_prefix) && !name.ends_with('/'))
            .filter_map(|name| archive.index_for_name(name))
            .collect();
        let mut data_size = 0u64;
        for index in indices {
            data_size = data_size.saturating_add(archive.by_index_raw(index)?.size());
        }
        Ok(data_size)
    }

    /// The `data.pkl` bytes. Refused beyond burn-pack's metadata ceiling, since a deflated
    /// entry can claim any decompressed size.
    pub fn pickle(&self) -> io::Result<Vec<u8>> {
        self.read_entry(&format!("{}data.pkl", self.root), MAX_METADATA_SIZE as u64)
    }

    /// A small text entry next to `data.pkl` (`version`, `byteorder`, ...), trimmed, if the
    /// archive has it.
    pub fn read_text(&self, name: &str) -> io::Result<Option<String>> {
        let full_name = format!("{}{name}", self.root);
        if !self.has_entry(name) {
            return Ok(None);
        }
        let bytes = self.read_entry(&full_name, MAX_TEXT_ENTRY_SIZE)?;
        let text = String::from_utf8(bytes)
            .map_err(|err| invalid_data(format!("ZIP entry '{full_name}': {err}")))?;
        Ok(Some(text.trim().to_string()))
    }

    /// Whether an entry next to `data.pkl` exists.
    pub fn has_entry(&self, name: &str) -> bool {
        let full_name = format!("{}{name}", self.root);
        lock_ignoring_poison(&self.archive)
            .index_for_name(&full_name)
            .is_some()
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let name = format!("{}data/{key}", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let mut entry = archive
            .by_name(&name)
            .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?;
        // A compressed entry yields its bytes only in order, so those before the window are
        // decompressed and dropped instead of being held.
        let size = entry.size();
        let skipped = (start as u64).min(size);
        io::copy(&mut (&mut entry).take(skipped), &mut io::sink())?;
        // Stopping short of the entry's end skips its CRC check; the bytes a tensor uses
        // are still validated against its declared extent.
        let rest = size - skipped;
        let bytes = read_zip_entry(
            &mut entry,
            &name,
            rest.min(max_len as u64),
            rest,
            self.file_len,
        )?;
        Ok((bytes, skipped as usize))
    }

    /// Read a whole entry that must not exceed `max_size` bytes.
    fn read_entry(&self, name: &str, max_size: u64) -> io::Result<Vec<u8>> {
        let mut archive = lock_ignoring_poison(&self.archive);
        let mut entry = archive
            .by_name(name)
            .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?;
        let size = entry.size();
        if size > max_size {
            return Err(invalid_data(format!(
                "ZIP entry '{name}' is {size} bytes, above the {max_size} byte limit"
            )));
        }
        read_zip_entry(&mut entry, name, size, size, self.file_len)
    }
}

/// Read the first `len` bytes of a ZIP entry whose header declares `size` bytes.
///
/// The `zip` crate checks an entry's CRC only when a read reaches its end, and a bounded
/// read stops on its own limit instead. When the whole entry is wanted, one read past
/// `size` reaches that end: a well-formed entry has nothing there and pays only for the
/// checksum, and bytes beyond the declared size mean the header lied about it.
fn read_zip_entry<R: Read>(
    entry: &mut R,
    name: &str,
    len: u64,
    size: u64,
    capacity_bound: usize,
) -> io::Result<Vec<u8>> {
    let bytes = read_exact_len(entry, len, capacity_bound)?;
    if len == size {
        let mut probe = [0u8; 1];
        let trailing = entry
            .read(&mut probe)
            .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?;
        if trailing != 0 {
            return Err(invalid_data(format!(
                "ZIP entry '{name}' holds more than the {size} bytes it declares"
            )));
        }
    }
    Ok(bytes)
}

/// The TAR container written by PyTorch before 0.1.10.
///
/// The `storages` entry is a count pickle followed, per storage, by a `(key, location,
/// storage type)` pickle, an `i64` element count and the raw bytes, then a list of storage
/// views. The whole entry is kept in memory and sliced. `reader::parse_tar_storages` parses
/// the layout, since that module has the pickle parser the metadata needs.
pub(crate) struct TarSource {
    blob: Vec<u8>,
    /// Storage key to `(offset, byte length)` within `blob`.
    layout: HashMap<String, (usize, usize)>,
}

impl TarSource {
    /// Wrap a parsed `storages` entry. Every range in `layout` must lie within `blob`.
    pub fn new(blob: Vec<u8>, layout: HashMap<String, (usize, usize)>) -> Self {
        Self { blob, layout }
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let &(offset, len) = self
            .layout
            .get(key)
            .ok_or_else(|| invalid_data(format!("storage '{key}' not found in TAR archive")))?;
        let start = start.min(len);
        let window = (len - start).min(max_len);
        offset
            .checked_add(start)
            .and_then(|from| Some(from..from.checked_add(window)?))
            .and_then(|range| self.blob.get(range))
            .map(|bytes| (bytes.to_vec(), start))
            .ok_or_else(|| invalid_data(format!("storage '{key}' lies outside the TAR data")))
    }
}

/// The legacy container (PyTorch 0.1.10 to 1.5, and `_use_new_zipfile_serialization=False`).
///
/// After the metadata pickles the file holds every storage back to back, in the order of a
/// key list that follows the main pickle. Each storage is an `i64` element count followed
/// by its bytes. The count prefix gives elements, not bytes, so the sizes declared by the
/// persistent ids in the main pickle are collected first and the layout is derived from
/// them once the key list is known.
pub(crate) struct LegacySource {
    path: PathBuf,
    state: Mutex<LegacyState>,
}

enum LegacyState {
    /// Collecting declarations while the main pickle is parsed:
    /// storage key to `(byte length, element size)`.
    Declaring(HashMap<String, (usize, usize)>),
    /// Layout fixed by the key list:
    /// storage key to `(file offset of the element count, byte length, element size)`.
    Finished(HashMap<String, (u64, usize, usize)>),
}

impl LegacySource {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            state: Mutex::new(LegacyState::Declaring(HashMap::new())),
        }
    }

    fn declare(&self, key: &str, byte_len: usize, element_size: usize) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        let LegacyState::Declaring(declared) = &mut *state else {
            return Err(invalid_data(format!(
                "storage '{key}' declared after the legacy layout was finalized"
            )));
        };
        match declared.get(key) {
            Some(&(prior_len, prior_size))
                if (prior_len, prior_size) != (byte_len, element_size) =>
            {
                Err(invalid_data(format!(
                    "storage '{key}' is declared twice with different sizes ({prior_len} and {byte_len} bytes)"
                )))
            }
            Some(_) => Ok(()),
            None => {
                declared.insert(key.to_string(), (byte_len, element_size));
                Ok(())
            }
        }
    }

    /// Fix the storage layout from the ordered key list and the data section bounds.
    pub fn finish(&self, keys: &[String], data_start: u64, file_len: u64) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        let LegacyState::Declaring(declared) = &*state else {
            return Err(invalid_data(
                "legacy storage layout finalized twice".to_string(),
            ));
        };
        let mut layout = HashMap::with_capacity(keys.len());
        let mut offset = data_start;

        for key in keys {
            let &(byte_len, element_size) = declared.get(key).ok_or_else(|| {
                invalid_data(format!(
                    "storage '{key}' is listed in the legacy key list but never referenced by the pickle"
                ))
            })?;
            let end = offset
                .checked_add(8)
                .and_then(|start| start.checked_add(byte_len as u64))
                .filter(|&end| end <= file_len)
                .ok_or_else(|| {
                    invalid_data(format!(
                        "storage '{key}' ({byte_len} bytes) extends beyond the end of the file"
                    ))
                })?;
            layout.insert(key.clone(), (offset, byte_len, element_size));
            offset = end;
        }

        *state = LegacyState::Finished(layout);
        Ok(())
    }

    fn read_storage(
        &self,
        key: &str,
        start: usize,
        max_len: usize,
    ) -> io::Result<(Vec<u8>, usize)> {
        let (offset, byte_len, element_size) = {
            let state = lock_ignoring_poison(&self.state);
            let LegacyState::Finished(layout) = &*state else {
                return Err(invalid_data(
                    "legacy storage layout was never finalized".to_string(),
                ));
            };
            *layout
                .get(key)
                .ok_or_else(|| invalid_data(format!("storage '{key}' not found in legacy file")))?
        };

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;

        let stored_numel = file.read_i64::<LittleEndian>()?;
        let expected_numel = (byte_len / element_size) as i64;
        if stored_numel != expected_numel {
            return Err(invalid_data(format!(
                "storage '{key}' holds {stored_numel} elements but the pickle declares {expected_numel}"
            )));
        }

        // `finish` checked that the storage lies within the file, so the length is trusted.
        let start = start.min(byte_len);
        file.seek(SeekFrom::Current(start as i64))?;
        let len = (byte_len - start).min(max_len);
        Ok((read_exact_len(&mut file, len as u64, len)?, start))
    }
}
