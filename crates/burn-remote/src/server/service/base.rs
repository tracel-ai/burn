//! Shared helpers for the connection handlers.

use crate::shared::{PROTOCOL_VERSION, RemoteMessage, SessionInit};

/// Decode the single `RemoteMessage::Init` a fresh submit (or fetch) socket opens with.
///
/// The handshake frame must hold exactly one message and it must be an `Init`; anything else is
/// a protocol error.
pub(crate) fn parse_init_handshake(bytes: &[u8]) -> Result<SessionInit, String> {
    let mut messages = rmp_serde::from_slice::<Vec<RemoteMessage>>(bytes)
        .map_err(|err| format!("Failed to decode init handshake: {err:?}"))?;

    match messages.pop() {
        Some(RemoteMessage::Init(init)) if messages.is_empty() => {
            if init.version != PROTOCOL_VERSION {
                return Err(format!(
                    "Unsupported Burn Remote protocol version {} (expected {PROTOCOL_VERSION})",
                    init.version
                ));
            }
            Ok(init)
        }
        other => Err(format!(
            "Init handshake expected a single RemoteMessage::Init, got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::SessionId;

    fn encode(messages: &[RemoteMessage]) -> Vec<u8> {
        rmp_serde::to_vec(messages).unwrap()
    }

    #[test]
    fn handshake_accepts_a_single_init() {
        let id = SessionId::new();
        let bytes = encode(&[RemoteMessage::Init(SessionInit::new(id, 3, vec![]))]);
        let init = parse_init_handshake(&bytes).unwrap();
        assert_eq!(init.session_id, id);
        assert_eq!(init.device_index, 3);
    }

    #[test]
    fn handshake_rejects_an_empty_batch() {
        assert!(parse_init_handshake(&encode(&[])).is_err());
    }

    #[test]
    fn handshake_rejects_more_than_one_message() {
        let id = SessionId::new();
        let bytes = encode(&[
            RemoteMessage::Init(SessionInit::new(id, 0, vec![])),
            RemoteMessage::Close(id),
        ]);
        assert!(parse_init_handshake(&bytes).is_err());
    }

    #[test]
    fn handshake_rejects_a_non_init() {
        let id = SessionId::new();
        assert!(parse_init_handshake(&encode(&[RemoteMessage::Close(id)])).is_err());
    }

    #[test]
    fn handshake_rejects_garbage() {
        assert!(parse_init_handshake(&[0xff, 0x00, 0x13]).is_err());
    }
}
