//! Where tensor bytes come from once a pickle has been parsed.
//!
//! Every PyTorch container keeps tensor metadata (a pickle) apart from the storages holding
//! the raw little-endian bytes. A [`StorageSource`] maps a storage key from that pickle to
//! its bytes on demand, so parsing a file costs nothing per tensor until its data is read.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zip::ZipArchive;
use zip::result::ZipError;

use super::reader::PytorchError;

/// Storage bytes for one container format.
pub(crate) enum StorageSource {
    Zip(ZipSource),
    Tar(TarSource),
    Legacy(LegacySource),
}

impl StorageSource {
    /// Read every byte of the storage saved under `key`.
    pub fn read(&self, key: &str) -> io::Result<Vec<u8>> {
        match self {
            Self::Zip(source) => source.read_storage(key),
            Self::Tar(source) => source.read_storage(key),
            Self::Legacy(source) => source.read_storage(key),
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

/// Read exactly `len` bytes without trusting `len` for an up-front allocation.
///
/// A length read from an untrusted file could be anything, so the buffer grows with the
/// bytes that actually arrive and a short read is an error rather than a huge allocation.
pub(crate) fn read_exact_len<R: Read>(reader: &mut R, len: u64) -> io::Result<Vec<u8>> {
    let mut buffer = Vec::new();
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
/// (`model/data.pkl`, `model/data/0`, `model/version`, ...). Older files use `archive/`,
/// and some tools write the entries at the root. The directory holding `data.pkl` is the
/// root for every other entry.
pub(crate) struct ZipSource {
    archive: Mutex<ZipArchive<BufReader<File>>>,
    /// Root directory including its trailing slash, or empty at the archive root.
    root: String,
    /// Total uncompressed size of the storage entries.
    data_size: u64,
}

impl ZipSource {
    pub fn open(path: &Path) -> Result<Self, PytorchError> {
        let archive = ZipArchive::new(BufReader::new(File::open(path)?))?;

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

        let data_prefix = format!("{root}data/");
        let mut data_size = 0u64;
        let mut archive = archive;
        for index in 0..archive.len() {
            let entry = archive.by_index_raw(index)?;
            if entry.name().starts_with(&data_prefix) && !entry.is_dir() {
                data_size = data_size.saturating_add(entry.size());
            }
        }

        Ok(Self {
            archive: Mutex::new(archive),
            root,
            data_size,
        })
    }

    /// Total uncompressed size of the storage entries.
    pub fn data_size(&self) -> u64 {
        self.data_size
    }

    /// The `data.pkl` bytes.
    pub fn pickle(&self) -> io::Result<Vec<u8>> {
        self.read_entry(&format!("{}data.pkl", self.root))
    }

    /// A small text entry next to `data.pkl` (`version`, `byteorder`, ...), trimmed, if the
    /// archive has it.
    pub fn read_text(&self, name: &str) -> io::Result<Option<String>> {
        let full_name = format!("{}{name}", self.root);
        let mut archive = lock_ignoring_poison(&self.archive);
        let entry = match archive.by_name(&full_name) {
            Ok(entry) => entry,
            Err(ZipError::FileNotFound) => return Ok(None),
            Err(err) => return Err(err.into()),
        };
        let mut text = String::new();
        entry.take(4096).read_to_string(&mut text)?;
        Ok(Some(text.trim().to_string()))
    }

    /// Whether an entry next to `data.pkl` exists.
    pub fn has_entry(&self, name: &str) -> bool {
        let full_name = format!("{}{name}", self.root);
        lock_ignoring_poison(&self.archive)
            .index_for_name(&full_name)
            .is_some()
    }

    fn read_storage(&self, key: &str) -> io::Result<Vec<u8>> {
        self.read_entry(&format!("{}data/{key}", self.root))
    }

    fn read_entry(&self, name: &str) -> io::Result<Vec<u8>> {
        let mut archive = lock_ignoring_poison(&self.archive);
        let mut entry = archive
            .by_name(name)
            .map_err(|err| invalid_data(format!("ZIP entry '{name}': {err}")))?;
        let size = entry.size();
        read_exact_len(&mut entry, size)
    }
}

/// The TAR container written by PyTorch before 0.1.10.
///
/// The `storages` entry is a count pickle followed, per storage, by a `(key, location,
/// storage type)` pickle, an `i64` element count and the raw bytes; the reader keeps the
/// whole entry in memory and slices it. Its layout is parsed by the reader, which also has
/// the pickle parser the metadata needs.
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

    /// Byte length of a storage, if the entry declares it.
    pub fn byte_len(&self, key: &str) -> Option<usize> {
        self.layout.get(key).map(|&(_, len)| len)
    }

    fn read_storage(&self, key: &str) -> io::Result<Vec<u8>> {
        let &(offset, len) = self
            .layout
            .get(key)
            .ok_or_else(|| invalid_data(format!("storage '{key}' not found in TAR archive")))?;
        self.blob
            .get(offset..offset + len)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| invalid_data(format!("storage '{key}' lies outside the TAR data")))
    }
}

/// The legacy container (PyTorch 0.1.10 to 1.5, and `_use_new_zipfile_serialization=False`).
///
/// After the metadata pickles the file holds every storage back to back, in the order of a
/// key list that follows the main pickle. Each storage is an `i64` element count followed
/// by its bytes. Nothing records where one storage ends, so the sizes declared by the
/// persistent ids in the main pickle are collected first and the layout is derived from
/// them once the key list is known.
pub(crate) struct LegacySource {
    path: PathBuf,
    state: Mutex<LegacyState>,
}

#[derive(Default)]
struct LegacyState {
    /// Storage key to `(byte length, element size)` as declared by the pickle.
    declared: HashMap<String, (usize, usize)>,
    /// Storage key to `(file offset of the element count, byte length, element size)`.
    layout: Option<HashMap<String, (u64, usize, usize)>>,
}

impl LegacySource {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            state: Mutex::new(LegacyState::default()),
        }
    }

    fn declare(&self, key: &str, byte_len: usize, element_size: usize) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        match state.declared.get(key) {
            Some(&(prior_len, prior_size))
                if (prior_len, prior_size) != (byte_len, element_size) =>
            {
                Err(invalid_data(format!(
                    "storage '{key}' is declared twice with different sizes ({prior_len} and {byte_len} bytes)"
                )))
            }
            Some(_) => Ok(()),
            None => {
                state
                    .declared
                    .insert(key.to_string(), (byte_len, element_size));
                Ok(())
            }
        }
    }

    /// Fix the storage layout from the ordered key list and the data section bounds.
    pub fn finish(&self, keys: &[String], data_start: u64, file_len: u64) -> io::Result<()> {
        let mut state = lock_ignoring_poison(&self.state);
        let mut layout = HashMap::with_capacity(keys.len());
        let mut offset = data_start;

        for key in keys {
            let &(byte_len, element_size) = state.declared.get(key).ok_or_else(|| {
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

        state.layout = Some(layout);
        Ok(())
    }

    fn read_storage(&self, key: &str) -> io::Result<Vec<u8>> {
        let (offset, byte_len, element_size) = {
            let state = lock_ignoring_poison(&self.state);
            let layout = state.layout.as_ref().ok_or_else(|| {
                invalid_data("legacy storage layout was never finalized".to_string())
            })?;
            *layout
                .get(key)
                .ok_or_else(|| invalid_data(format!("storage '{key}' not found in legacy file")))?
        };

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut count = [0u8; 8];
        file.read_exact(&mut count)?;
        let stored_numel = i64::from_le_bytes(count);
        let expected_numel = (byte_len / element_size) as i64;
        if stored_numel != expected_numel {
            return Err(invalid_data(format!(
                "storage '{key}' holds {stored_numel} elements but the pickle declares {expected_numel}"
            )));
        }

        read_exact_len(&mut file, byte_len as u64)
    }
}
