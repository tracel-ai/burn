use std::str::FromStr;

use burn::server::RemoteSecret;
use burn::tensor::{Device, DeviceType};
use iroh::EndpointId;

/// The devices to draw from one compute server, parsed from a `--server` argument.
///
/// `ws://host:port#0,2` selects devices 0 and 2 of the WebSocket server at that address;
/// without a fragment, every device the server hosts is used. `iroh://topic#0,2` is the
/// same over Iroh, except that a fragment is required there: device enumeration is only
/// exposed through the facade for WebSocket servers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerDevices {
    pub transport: Transport,
    pub selection: Selection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// A full WebSocket address, e.g. `ws://192.168.2.139:3000`.
    WebSocket(String),
    /// A topic string both sides derive the server identity from; see [`topic_secret`].
    Iroh(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// Every device the server hosts.
    Hosted,
    /// Specific device indices on the server.
    Indices(Vec<usize>),
}

impl FromStr for ServerDevices {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let (address, fragment) = match input.split_once('#') {
            Some((address, fragment)) => (address, Some(fragment)),
            None => (input, None),
        };

        let selection = match fragment {
            None => Selection::Hosted,
            Some(fragment) => {
                let indices = fragment
                    .split(',')
                    .map(|index| {
                        index
                            .trim()
                            .parse::<usize>()
                            .map_err(|_| format!("invalid device index `{index}` in `{input}`"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if indices.is_empty() {
                    return Err(format!("empty device list in `{input}`"));
                }
                Selection::Indices(indices)
            }
        };

        let transport = if address.starts_with("ws://") {
            Transport::WebSocket(address.to_string())
        } else if let Some(topic) = address.strip_prefix("iroh://") {
            if topic.is_empty() {
                return Err(format!("empty iroh topic in `{input}`"));
            }
            Transport::Iroh(topic.to_string())
        } else {
            return Err(format!(
                "`{input}` must start with `ws://` or `iroh://` (e.g. `ws://host:3000#0,1` \
                 or `iroh://my-topic#0`)"
            ));
        };

        Ok(Self {
            transport,
            selection,
        })
    }
}

impl ServerDevices {
    /// Connect to the server and return one unified [`Device`] per selected device.
    ///
    /// `endpoint` carries the caller's Iroh endpoint and must be `Some` for an Iroh server.
    ///
    /// # Panics
    ///
    /// Panics for an Iroh server without an explicit device list: the facade has no Iroh
    /// enumeration, so the indices must come from the spec.
    pub fn connect(&self, endpoint: Option<&iroh::Endpoint>) -> Vec<Device> {
        match (&self.transport, &self.selection) {
            (Transport::WebSocket(address), Selection::Hosted) => {
                Device::enumerate(DeviceType::remote_websocket(address)).into_vec()
            }
            (Transport::WebSocket(address), Selection::Indices(indices)) => indices
                .iter()
                .map(|index| Device::remote_websocket(address, *index))
                .collect(),
            (Transport::Iroh(topic), Selection::Indices(indices)) => {
                let endpoint = endpoint.expect("An Iroh spec binds an endpoint before connecting");
                let server: EndpointId = topic_secret(topic).id();
                indices
                    .iter()
                    .map(|index| Device::remote_iroh(endpoint, server, *index))
                    .collect()
            }
            (Transport::Iroh(topic), Selection::Hosted) => panic!(
                "Device enumeration is not exposed for Iroh servers; list the indices \
                 explicitly, e.g. `iroh://{topic}#0,1,2,3`"
            ),
        }
    }

    /// Whether connecting needs a bound Iroh endpoint.
    pub fn needs_endpoint(&self) -> bool {
        matches!(self.transport, Transport::Iroh(_))
    }
}

/// Derive the server identity from a shared topic string.
///
/// Both sides agree on an address without exchanging keys, at the cost that anyone who
/// knows the topic can host under this identity. A real deployment uses
/// `RemoteSecret::random()` and distributes its public id instead.
pub fn topic_secret(topic: &str) -> RemoteSecret {
    let hash = blake3::hash(format!("burn-remote-training:{topic}").as_bytes());
    RemoteSecret::from_bytes(*hash.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_websocket_with_indices() {
        let spec: ServerDevices = "ws://192.168.2.139:3000#0,2".parse().unwrap();
        assert_eq!(
            spec,
            ServerDevices {
                transport: Transport::WebSocket("ws://192.168.2.139:3000".to_string()),
                selection: Selection::Indices(vec![0, 2]),
            }
        );
    }

    #[test]
    fn parses_websocket_without_fragment_as_hosted() {
        let spec: ServerDevices = "ws://localhost:3000".parse().unwrap();
        assert_eq!(spec.selection, Selection::Hosted);
    }

    #[test]
    fn parses_iroh_topic() {
        let spec: ServerDevices = "iroh://my-rig#1".parse().unwrap();
        assert_eq!(
            spec,
            ServerDevices {
                transport: Transport::Iroh("my-rig".to_string()),
                selection: Selection::Indices(vec![1]),
            }
        );
    }

    #[test]
    fn rejects_unknown_scheme() {
        assert!("http://localhost:3000".parse::<ServerDevices>().is_err());
    }

    #[test]
    fn rejects_bad_index() {
        assert!("ws://localhost:3000#a".parse::<ServerDevices>().is_err());
    }
}
