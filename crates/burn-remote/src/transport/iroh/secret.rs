//! Server identity for the Iroh transport.

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
    /// across restarts. Whoever can read the file can pose as the server, so it is created
    /// readable by its owner only.
    #[cfg(not(target_family = "wasm"))]
    pub fn load_or_create(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind, Write};

        let path = path.as_ref();
        match std::fs::read(path) {
            Ok(bytes) => bytes.try_into().map(Self::from_bytes).map_err(|_| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("{} is not a 32-byte identity", path.display()),
                )
            }),
            Err(err) if err.kind() == ErrorKind::NotFound => {
                let secret = Self::random();
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
                options.open(path)?.write_all(&secret.to_bytes())?;
                Ok(secret)
            }
            Err(err) => Err(err),
        }
    }

    /// The public identity clients dial.
    pub fn id(&self) -> iroh::EndpointId {
        self.0.public()
    }

    #[cfg(feature = "server")]
    pub(crate) fn secret_key(&self) -> iroh::SecretKey {
        self.0.clone()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn a_created_identity_is_loaded_back() {
        let dir = std::env::temp_dir().join(format!("burn-remote-secret-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("server.key");

        let created = RemoteSecret::load_or_create(&path).unwrap();
        let loaded = RemoteSecret::load_or_create(&path).unwrap();
        assert_eq!(created.id(), loaded.id());

        std::fs::write(&path, b"short").unwrap();
        assert!(RemoteSecret::load_or_create(&path).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
