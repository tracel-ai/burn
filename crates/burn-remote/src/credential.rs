use core::fmt;
use std::sync::Arc;

/// What a client presents to a server's authorizer, such as the token a `TokenAuthorizer` checks.
///
/// Its `Debug` output never shows the bytes, since a credential is often a shared secret.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Credential(Arc<[u8]>);

impl Credential {
    /// The credential's bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Whether the credential is empty, which is what a client that sets none presents.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether the credential equals `expected`, in time that depends only on their lengths, so an
    /// authorizer does not leak how much of a guess was right.
    pub fn matches(&self, expected: &[u8]) -> bool {
        if self.0.len() != expected.len() {
            return false;
        }
        self.0
            .iter()
            .zip(expected)
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Credential(..)")
    }
}

impl From<&[u8]> for Credential {
    fn from(bytes: &[u8]) -> Self {
        Self(bytes.into())
    }
}

impl From<Vec<u8>> for Credential {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes.into())
    }
}

impl From<&str> for Credential {
    fn from(token: &str) -> Self {
        Self(token.as_bytes().into())
    }
}

impl From<String> for Credential {
    fn from(token: String) -> Self {
        Self(token.into_bytes().into())
    }
}

impl From<&String> for Credential {
    fn from(token: &String) -> Self {
        token.as_str().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_credential_matches_only_its_own_bytes() {
        let credential = Credential::from("secret");
        assert!(credential.matches(b"secret"));
        assert!(!credential.matches(b"secreT"));
        assert!(!credential.matches(b"secret!"));
    }

    #[test]
    fn a_credential_stays_out_of_debug_output() {
        assert_eq!(
            format!("{:?}", Credential::from("secret")),
            "Credential(..)"
        );
    }
}
