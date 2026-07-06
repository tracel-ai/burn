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

    /// The public identity clients dial.
    pub fn id(&self) -> iroh::EndpointId {
        self.0.public()
    }

    #[cfg(feature = "server")]
    pub(crate) fn secret_key(&self) -> iroh::SecretKey {
        self.0.clone()
    }
}
