use burn::server::{Channel, RemoteSecret};
use burn::tensor::{Device, Distribution, Tensor};
use iroh::{Endpoint, EndpointId, endpoint::presets};
use tracing_subscriber::{EnvFilter, fmt};

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,burn_remote=debug"));
    fmt().with_env_filter(filter).init();
}

/// Derive a stable server identity from a human-friendly topic, so both ends agree on the address
/// without exchanging keys. The topic acts as a shared secret here (anyone who knows it can host as
/// this identity), which suits a demo; a real deployment would use `RemoteSecret::random()` and
/// share its `id()`.
fn topic_secret(topic: &str) -> RemoteSecret {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    RemoteSecret::from_bytes(*hash.as_bytes())
}

pub async fn run_server(topic: &str) {
    init_logging();
    let secret = topic_secret(topic);
    tracing::info!(topic, server_id = %secret.id(), "server ready");
    tracing::info!("waiting for clients (press Ctrl-C to stop)");
    burn::server::start_async(
        Device::flex(),
        Channel::Iroh {
            secret: Box::new(secret),
        },
    )
    .await;
    tracing::info!("server stopped");
}

pub async fn run_client(topic: &str) {
    let server_id: EndpointId = topic_secret(topic).id();

    println!("topic     : {topic}");
    println!("server id : {server_id}");
    println!("connecting...");

    let endpoint = Endpoint::builder(presets::N0)
        .bind()
        .await
        .expect("bind failed");
    let device = Device::remote_iroh(&endpoint, server_id, 0);

    println!("connected\n");
    train(&device);
}

fn train(device: &Device) {
    const N: usize = 512;
    const STEPS: usize = 80;
    const LR: f32 = 0.08;

    println!("target: y = 2.5 * x + 0.5");
    println!("steps : {STEPS}  samples: {N}\n");

    let x = Tensor::<1>::random([N], Distribution::Default, device) * 2.0 - 1.0;
    let y_true = x.clone() * 2.5 + 0.5;

    let mut w = Tensor::<1>::from_floats([0.0_f32], device);
    let mut b = Tensor::<1>::from_floats([0.0_f32], device);

    println!("{:>5}  {:>10}", "step", "loss");

    for step in 0..STEPS {
        let y_pred = x.clone() * w.clone().expand([N]) + b.clone().expand([N]);
        let error = y_pred - y_true.clone();
        let loss = (error.clone() * error.clone()).mean();
        let dw = (x.clone() * error.clone()).mean() * 2.0_f32;
        let db = error.mean() * 2.0_f32;

        w = w - dw * LR;
        b = b - db * LR;

        if step % 10 == 0 || step == STEPS - 1 {
            let loss_val = loss.to_data().to_vec::<f32>().unwrap()[0];
            println!("{:>5}  {:>10.6}", step + 1, loss_val);
        }
    }

    let w_val = w.to_data().to_vec::<f32>().unwrap()[0];
    let b_val = b.to_data().to_vec::<f32>().unwrap()[0];

    println!("\nlearned: y = {w_val:.4} * x + {b_val:.4}");
    println!("target : y = 2.5000 * x + 0.5000");
}
