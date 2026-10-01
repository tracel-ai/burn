//! Server identity for the Iroh transport.

#[cfg(not(target_family = "wasm"))]
use std::{
    fs,
    io::{self, ErrorKind, Write},
    path::Path,
};

#[cfg(not(target_family = "wasm"))]
use tempfile::NamedTempFile;

/// A compute server's stable identity: the secret stays on the server, and the public
/// [`id`](Self::id) it yields is the address clients dial. Generate one with [`random`](Self::random)
/// and persist [`to_bytes`](Self::to_bytes) for a stable address across restarts, or derive it from a
/// seed with [`from_bytes`](Self::from_bytes).
#[derive(Clone)]
pub struct RemoteSecret(iroh::SecretKey);

impl RemoteSecret {
    /// A fresh random identity. Persist [`to_bytes`](Self::to_bytes) to reuse the same address later.
    pub fn random() -> Self {
        Self(iroh::SecretKey::generate())
    }

    /// A deterministic identity from 32 seed bytes (e.g. a hash of an application name).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(iroh::SecretKey::from_bytes(&bytes))
    }

    /// The raw 32 bytes, to persist and reload a stable identity.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    /// The identity stored in `path`, created there on first use so the server keeps its id
    /// across restarts. Whoever can read the file can pose as the server: on Unix it is created
    /// readable by its owner only, elsewhere it takes its directory's permissions.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_or_create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        match Self::load(path) {
            Err(err) if err.kind() == ErrorKind::NotFound => Self::create(path),
            loaded => loaded,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    fn load(path: &Path) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        bytes.try_into().map(Self::from_bytes).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("{} is not a 32-byte identity", path.display()),
            )
        })
    }

    /// Written aside and moved into place, so a crash never leaves a partial key behind and a
    /// key another process created meanwhile is kept.
    #[cfg(not(target_family = "wasm"))]
    fn create(path: &Path) -> io::Result<Self> {
        let dir = match path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir,
            _ => Path::new("."),
        };
        let secret = Self::random();
        let mut staged = NamedTempFile::new_in(dir)?;
        staged.write_all(&secret.to_bytes())?;
        staged.as_file().sync_all()?;
        match staged.persist_noclobber(path) {
            Ok(_) => Ok(secret),
            Err(err) if err.error.kind() == ErrorKind::AlreadyExists => Self::load(path),
            Err(err) => Err(err.error),
        }
    }

    /// The public identity clients dial.
    pub fn id(&self) -> iroh::EndpointId {
        self.0.public()
    }

    #[cfg(all(feature = "server", not(target_family = "wasm")))]
    pub(crate) fn secret_key(&self) -> iroh::SecretKey {
        self.0.clone()
    }
}

impl core::fmt::Debug for RemoteSecret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RemoteSecret")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn a_created_identity_is_loaded_back() {
        let dir = scratch_dir("loaded-back");
        let path = dir.join("server.key");

        let created = RemoteSecret::load_or_create(&path).unwrap();
        let loaded = RemoteSecret::load_or_create(&path).unwrap();
        assert_eq!(created.id(), loaded.id());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);

        std::fs::write(&path, b"short").unwrap();
        assert!(RemoteSecret::load_or_create(&path).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn concurrent_creators_agree_on_one_identity() {
        let dir = scratch_dir("concurrent");
        let path = dir.join("server.key");

        let ids: Vec<_> = std::thread::scope(|scope| {
            let creators: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| RemoteSecret::load_or_create(&path).unwrap().id()))
                .collect();
            creators.into_iter().map(|c| c.join().unwrap()).collect()
        });
        assert!(ids.iter().all(|id| *id == ids[0]));
        assert_eq!(RemoteSecret::load_or_create(&path).unwrap().id(), ids[0]);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_created_identity_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("owner-only");
        let path = dir.join("server.key");
        RemoteSecret::load_or_create(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn scratch_dir(test: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("burn-remote-secret-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
