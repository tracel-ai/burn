//! The compute environments of a [`CubeBackend`](crate::CubeBackend) device, in
//! cubecl's environment storage.
//!
//! Cubecl keeps each environment as one database in its cache root, and one
//! environment active for the whole process: activating one for a device
//! activates it for every device of the process. What a device compiles and
//! tunes is kept under its hardware inside the environment, so devices sharing
//! one never read each other's entries.

use burn_backend::{EnvironmentBundle, EnvironmentError, EnvironmentId, EnvironmentOrigin};
use burn_std::Bytes;
use cubecl::bundle::{BundleFormat, ExportOptions, export};
use cubecl::environment;
use cubecl_environment::persistence::{Access, Database, Origin};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// How many entries one write into an environment carries.
const BATCH: usize = 256;

/// The environment storage of this process: cubecl's cache root.
pub(crate) struct EnvironmentStorage {
    /// What tells this process's storage from every other's, a server's
    /// included: random, drawn once.
    identity: u64,
    root: PathBuf,
}

impl EnvironmentStorage {
    /// The storage as cubecl's configuration places it now.
    pub(crate) fn current() -> Self {
        static IDENTITY: OnceLock<u64> = OnceLock::new();
        let identity = *IDENTITY.get_or_init(|| {
            use core::hash::BuildHasher;
            std::collections::hash_map::RandomState::new().hash_one(std::process::id())
        });
        Self {
            identity,
            root: environment::root(),
        }
    }

    /// Make the environment `name` hold what `origin` says, replacing what it
    /// held.
    ///
    /// Emptied entry by entry rather than deleted: cubecl keeps one connection
    /// per database for the life of the process, so a file replaced under it
    /// would leave that connection writing into one nothing reads. Everything
    /// that could refuse is checked before it is emptied.
    pub(crate) fn create(
        &self,
        name: String,
        origin: EnvironmentOrigin,
    ) -> Result<EnvironmentId, EnvironmentError> {
        if *environment::active() == *name {
            return Err(EnvironmentError::Active { name });
        }
        let staged = self.staged(&name, "bundle");
        let created = self
            .source(origin, &staged)
            .and_then(|source| self.replace(&name, source.as_ref()));
        remove(&staged);
        created.map(|_| self.id(name))
    }

    /// What a new environment is copied from, opened: `None` for an empty one.
    /// A bundle is written to `staged` to be read.
    fn source(
        &self,
        origin: EnvironmentOrigin,
        staged: &Path,
    ) -> Result<Option<Database>, EnvironmentError> {
        match origin {
            EnvironmentOrigin::Empty => Ok(None),
            EnvironmentOrigin::Fork(source) => {
                self.reaches(&source)?;
                self.open(&source.name, Access::ReadOnly).map(Some)
            }
            EnvironmentOrigin::Bundle(bundle) => {
                std::fs::write(staged, &*bundle.bytes).map_err(storage)?;
                Database::open_at(staged, Access::ReadOnly)
                    .map(Some)
                    .map_err(|reason| EnvironmentError::Storage { reason })
            }
        }
    }

    /// Empty the environment `name`, then copy `source` into it.
    fn replace(&self, name: &str, source: Option<&Database>) -> Result<(), EnvironmentError> {
        let created = self.open(name, Access::ReadWrite)?;
        for namespace in created.namespaces() {
            created.purge(&namespace);
        }
        if let Some(source) = source {
            copy(source, &created);
        }
        Ok(())
    }

    /// Make `id` the process's active environment, answering the one before.
    pub(crate) fn activate(&self, id: EnvironmentId) -> Result<EnvironmentId, EnvironmentError> {
        self.reaches(&id)?;
        let previous = self.active();
        environment::activate(&id.name);
        Ok(previous)
    }

    /// The process's active environment.
    pub(crate) fn active(&self) -> EnvironmentId {
        self.id(environment::active().to_string())
    }

    /// The environment `id`, serialized whole.
    pub(crate) fn save(&self, id: EnvironmentId) -> Result<EnvironmentBundle, EnvironmentError> {
        self.reaches(&id)?;
        let staged = self.staged(&id.name, "saved");
        let options = ExportOptions {
            name: id.name.clone(),
            format: BundleFormat::Sqlite,
            ..Default::default()
        };
        let saved = export(&[self.path(&id.name)], &staged, &options)
            .map_err(|error| storage(error.to_string()))
            .and_then(|_| std::fs::read(&staged).map_err(storage));
        remove(&staged);
        Ok(EnvironmentBundle {
            bytes: Bytes::from_bytes_vec(saved?),
        })
    }

    fn id(&self, name: String) -> EnvironmentId {
        EnvironmentId {
            storage: self.identity,
            name,
        }
    }

    /// Refuses an environment another storage keeps.
    fn reaches(&self, id: &EnvironmentId) -> Result<(), EnvironmentError> {
        match id.storage == self.identity {
            true => Ok(()),
            false => Err(EnvironmentError::IncompatibleDevice {
                reason: alloc::format!(
                    "the environment {} lives in another process's storage",
                    id.name
                ),
            }),
        }
    }

    fn open(&self, name: &str, access: Access) -> Result<Database, EnvironmentError> {
        Database::open_at(&self.path(name), access)
            .map_err(|reason| EnvironmentError::Storage { reason })
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(environment::file_name(name))
    }

    /// A scratch file beside the environment `name`, for `purpose`.
    fn staged(&self, name: &str, purpose: &str) -> PathBuf {
        let mut path = self.path(name);
        path.set_extension(alloc::format!("{purpose}.staged"));
        path
    }
}

/// Copy every entry of `source` into `target`, a namespace at a time and in
/// batches, so no write holds more than a batch.
fn copy(source: &Database, target: &Database) {
    for namespace in source.namespaces() {
        let mut entries = Vec::new();
        source.scan(&namespace, &mut |key, value| {
            entries.push((
                Bytes::from_bytes_vec(key.to_vec()),
                Bytes::from_bytes_vec(value.to_vec()),
            ));
        });
        let mut entries = entries.into_iter().peekable();
        while entries.peek().is_some() {
            let mut batch = entries.by_ref().take(BATCH);
            target.insert_many(&namespace, &mut batch, Origin::Imported);
        }
    }
}

/// Delete a scratch file and the files `SQLite` writes beside it.
fn remove(path: &Path) {
    for suffix in ["", "-wal", "-shm", "-tshm"] {
        let mut file = path.as_os_str().to_owned();
        file.push(suffix);
        let _ = std::fs::remove_file(file);
    }
}

fn storage(error: impl ToString) -> EnvironmentError {
    EnvironmentError::Storage {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An environment another storage keeps — another process's, a server's —
    /// is refused before anything is read, written or activated.
    #[test]
    fn an_environment_of_another_storage_is_incompatible() {
        let storage = EnvironmentStorage::current();
        let foreign = EnvironmentId {
            storage: storage.identity.wrapping_add(1),
            name: "foreign".into(),
        };

        assert!(matches!(
            storage.activate(foreign.clone()),
            Err(EnvironmentError::IncompatibleDevice { .. })
        ));
        assert!(matches!(
            storage.save(foreign.clone()),
            Err(EnvironmentError::IncompatibleDevice { .. })
        ));
        assert!(matches!(
            storage.create("forked".into(), EnvironmentOrigin::Fork(foreign)),
            Err(EnvironmentError::IncompatibleDevice { .. })
        ));
    }
}
