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
/// The destination is fixed at [`create`](Self::create) and the scratch path is a sibling of
/// it, so [`commit`](Self::commit) stays on one filesystem: never `EXDEV`, and atomic on
/// POSIX. (Windows documents no atomicity guarantee when the rename replaces an existing
/// file.)
///
/// Publishing replaces the destination rather than truncating it. Its permission bits are
/// carried over on Unix, but its ownership and hard links cannot be: the published file is a
/// new inode.
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
/// scratch.commit()?;
/// # Ok(())
/// # }
/// ```
pub struct AtomicFile {
    path: PathBuf,
    destination: PathBuf,
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
    /// The guard keeps `destination`, so [`commit`](Self::commit) can only publish to the
    /// path the scratch file was placed beside. That is what makes the same-filesystem
    /// rename a property of the type rather than a contract the caller has to honour.
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
                            destination: destination.to_path_buf(),
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

    /// Flush the completed file to disk and move it onto the destination.
    ///
    /// For a caller that wrote through a handle it opened itself and has since closed. The
    /// scratch file is reopened here only to sync it, which is what orders the data before
    /// the rename: a crash must not leave the destination pointing at bytes that never
    /// reached the platter. The reopen asks for write access even though nothing is written
    /// through it, because Windows backs `sync_all` with `FlushFileBuffers`, which returns
    /// `Access is denied` on a read-only handle.
    ///
    /// A caller that still holds the handle it wrote through should sync and drop that
    /// handle first: a deferred write error (NFS over quota, a failing disk) is reported to
    /// the handle that wrote the bytes and is lost once that handle is closed, so the sync
    /// here cannot stand in for it.
    pub fn commit(mut self) -> Result<(), Error> {
        File::options()
            .write(true)
            .open(&self.path)
            .and_then(|file| file.sync_all())
            .map_err(|e| {
                Error::IoError(format!(
                    "cannot flush '{}' to disk: {e}",
                    self.path.display()
                ))
            })?;

        self.rename_onto()
    }

    /// Give the scratch file the permission bits of the file it is about to replace.
    ///
    /// Publishing by rename hands the destination a brand new inode, which without this
    /// carries the process umask: a `0600` checkpoint re-saved under umask `022` would come
    /// back `0644`, widening access to the model that was deliberately kept private. Copying
    /// the mode across restores what a truncating overwrite would have left. Ownership and
    /// hard links have no such answer - a new inode cannot keep them - and Windows has no
    /// mode to copy, its `Permissions` carrying only a read-only flag.
    ///
    /// Nothing is copied when the destination does not exist (there is no prior mode, so the
    /// umask is the right answer) or is not a regular file (the rename replaces a symlink
    /// rather than following it, so the link target's mode is not the one being replaced).
    ///
    /// A failure here fails the save rather than publishing at the umask, since quietly
    /// widening a checkpoint's permissions is the outcome this exists to prevent. It is
    /// reported before the rename, so the destination still holds the old file.
    #[cfg(unix)]
    fn inherit_permissions(&self) -> Result<(), Error> {
        let metadata = match std::fs::symlink_metadata(&self.destination) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => return Ok(()),
        };

        std::fs::set_permissions(&self.path, metadata.permissions()).map_err(|e| {
            Error::IoError(format!(
                "cannot carry the permissions of '{}' onto the file replacing it: {e}",
                self.destination.display()
            ))
        })
    }

    /// Move the completed file onto the destination, replacing any existing file there.
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
    pub(crate) fn rename_onto(&mut self) -> Result<(), Error> {
        #[cfg(unix)]
        self.inherit_permissions()?;

        std::fs::rename(&self.path, &self.destination).map_err(|e| {
            Error::IoError(format!(
                "cannot move the completed container '{}' onto '{}': {e}",
                self.path.display(),
                self.destination.display()
            ))
        })?;
        self.persisted = true;

        #[cfg(unix)]
        {
            // An empty parent means a bare relative file name; the directory is the cwd.
            let parent = match self.destination.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent,
                _ => Path::new("."),
            };
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|e| {
                    Error::IoError(format!(
                        "'{}' was saved, but syncing its directory failed, so the rename may \
                         not survive power loss: {e}",
                        self.destination.display()
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
