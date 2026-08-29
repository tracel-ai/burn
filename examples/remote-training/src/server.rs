use burn::server::Channel;
use burn::tensor::Device;

use crate::spec::{Selection, ServerDevices, Transport, topic_secret};

/// Host the compiled backend's devices for remote clients, blocking until shutdown.
///
/// The server always hosts every device the backend enumerates; a `#` device list on the
/// listen spec is a client-side concept and is rejected here to avoid suggesting otherwise.
pub fn serve(spec: &ServerDevices) {
    if !matches!(spec.selection, Selection::Hosted) {
        panic!(
            "The server hosts every device of its backend; select devices on the client's \
             `--server` argument instead"
        );
    }

    let channel = match &spec.transport {
        Transport::WebSocket(address) => Channel::WebSocket {
            port: port_of(address),
        },
        Transport::Iroh(topic) => {
            let secret = topic_secret(topic);
            tracing::info!(topic, server_id = %secret.id(), "iroh identity");
            Channel::Iroh {
                secret: Box::new(secret),
            }
        }
    };

    burn::server::start(Device::default(), channel);
}

fn port_of(address: &str) -> u16 {
    address
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse().ok())
        .unwrap_or_else(|| panic!("no port in `{address}`; expected e.g. `ws://0.0.0.0:3000`"))
}
