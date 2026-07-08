#![recursion_limit = "141"]

use burn::{
    server::{Channel, RemoteSecret},
    tensor::Device,
};

/// Derive a stable server identity from a shared topic string. The client derives the same identity
/// from the same topic, so it can reach this server over Iroh with no IP address or port
/// forwarding. A real deployment would use `RemoteSecret::random()` and share its `id()` instead.
fn topic_secret(topic: &str) -> RemoteSecret {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    RemoteSecret::from_bytes(*hash.as_bytes())
}

pub fn start() {
    let topic = std::env::var("REMOTE_TOPIC").unwrap_or_else(|_| "db-pedia-train-default".into());
    let secret = topic_secret(&topic);

    println!("topic     : {topic}");
    println!("server id : {}", secret.id());
    println!("waiting for clients (Ctrl-C to stop)");

    let runtime = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    runtime.block_on(async {
        burn::server::start_async(
            Device::default(),
            Channel::Iroh {
                secret: Box::new(secret),
            },
        )
        .await;
    });
}
