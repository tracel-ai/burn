//! Writing a file without ever leaving a partial one at its destination.
//!
//! [`Writer::write_to_file_atomic`](crate::Writer::write_to_file_atomic) is built on this, and
//! it is public because burn-pack is not the only crate that writes a model file whose bytes
//! are produced during the write. A save whose tensors materialize on demand can fail partway
//! through for reasons that have nothing to do with the disk, and must not destroy the
//! checkpoint that was already there.

use alloc::format;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::base::Error;

/// A scratch file next to an eventual destination, deleted unless it is persisted.
///
/// Gives a write its all-or-nothing behaviour: the file is built here and only takes the
/// destination's name once it is complete. Any earlier return drops the guard, which removes
/// the partial file.
///
/// The scratch path is a sibling of the destination rather than a system temp file so that
/// [`commit`](Self::commit) stays on one filesystem: never `EXDEV`, and atomic on POSIX.
/// (Windows documents no atomicity guarantee when the rename replaces an existing file.)
///
/// # Example
///
/// ```no_run
/// # use burn_pack::AtomicFile;
/// # fn write_my_format(_path: &std::path::Path) -> Result<(), burn_pack::Error> { Ok(()) }
/// # fn main() -> Result<(), burn_pack::Error> {
/// let destination = std::path::Path::new("model.myfmt");
///
/// // The handle is dropped here because this writer opens the path itself; keep it instead
/// // when you mean to write through it.
/// let (scratch, _reserved) = AtomicFile::create(destination)?;
/// write_my_format(scratch.path())?;
/// scratch.commit(destination)?;
/// # Ok(())
/// # }
/// ```
pub struct AtomicFile {
    path: PathBuf,
    persisted: bool,
}

impl AtomicFile {
    /// Create a scratch file alongside `destination` and hand back a handle onto it.
    ///
    /// The guard and its file are handed back together so neither can exist without the
    /// other: no path is reserved that nothing will clean up, and no file is created that no
    /// guard is watching. A caller that writes through its own handle (because the library
    /// doing the writing opens the path itself) can drop the returned one; the exclusive
    /// create has already done its job by then.
    ///
    /// The file is opened with `create_new`, which refuses a path that already exists and
    /// does not follow a symlink planted there - a plain `File::create` would truncate
    /// whatever such a link points at. The scratch name is predictable (pid + counter), so a
    /// stale leftover from a killed run, pid reuse, or a deliberately pre-created link all
    /// surface as `AlreadyExists`, answered by moving on to the next counter value.
    pub fn create(destination: &Path) -> Result<(Self, File), Error> {
        loop {
            let path = Self::path_beside(destination)?;
            match File::create_new(&path) {
                Ok(file) => {
                    return Ok((
                        Self {
                            path,
                            persisted: false,
                        },
                        file,
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

    /// Where to write. Nothing may be written to the destination itself until
    /// [`commit`](Self::commit).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Pick a scratch path alongside `destination`.
    ///
    /// The name carries the process id and a counter so concurrent writers for the same
    /// destination pick distinct names; [`create`](Self::create)'s exclusive open is what
    /// makes the rare collision harmless rather than this scheme's uniqueness.
    fn path_beside(destination: &Path) -> Result<PathBuf, Error> {
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

    /// Flush the completed file to disk and move it onto `destination`.
    ///
    /// For a caller that wrote through a handle it opened itself and has since closed. The
    /// scratch file is reopened here only to sync it, which is what orders the data before
    /// the rename: a crash must not leave the destination pointing at bytes that never
    /// reached the platter.
    ///
    /// A caller that still holds the handle it wrote through should sync that instead and go
    /// through [`rename_onto`](Self::rename_onto), since a deferred write error (NFS over
    /// quota, a failing disk) is reported to the handle that wrote it and is lost once that
    /// handle is closed.
    pub fn commit(mut self, destination: &Path) -> Result<(), Error> {
        File::open(&self.path)
            .and_then(|file| file.sync_all())
            .map_err(|e| {
                Error::IoError(format!(
                    "cannot flush '{}' to disk: {e}",
                    self.path.display()
                ))
            })?;

        self.rename_onto(destination)
    }

    /// Move the completed file onto `destination`, replacing any existing file there.
    ///
    /// Assumes the data is already durable; [`commit`](Self::commit) is the form that makes
    /// sure of it.
    ///
    /// On Unix the parent directory is synced after the rename, making the rename itself
    /// durable: once this returns `Ok`, power loss cannot revert the destination to the old
    /// file. (Windows has no portable directory sync; NTFS journals metadata on its own
    /// schedule.) If that final sync fails, the error says so explicitly - the new file is at
    /// the destination and intact, only its durability is unconfirmed - because the generic
    /// rename error below would wrongly imply the save did not happen.
    ///
    /// On rename failure the finished file is deleted by the guard, so the error names it:
    /// the bytes were written and then discarded, which is worth saying out loud.
    pub(crate) fn rename_onto(&mut self, destination: &Path) -> Result<(), Error> {
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

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if !self.persisted {
            // Best effort: the write already failed, and a leftover scratch file is a less
            // useful thing to report than whatever went wrong.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
