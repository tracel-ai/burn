#![recursion_limit = "256"]

use burn::tensor::Device;
use burn::tensor::distributed::{DistributedConfig, ReduceOperation};
use burn::train::{ExecutionStrategy, MultiDeviceOptim};
use clap::{Parser, Subcommand, ValueEnum};
use iroh::{Endpoint, endpoint::presets};
use tracing_subscriber::{EnvFilter, fmt};

use remote_training::spec::ServerDevices;
use remote_training::training::TrainConfig;
use remote_training::{server, training};

#[derive(Parser)]
#[command(about = "MNIST training on the devices of one or more burn remote servers")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Host every device of the compiled backend (`ws://0.0.0.0:3000` or `iroh://topic`).
    Server { listen: ServerDevices },
    /// Train MNIST data-parallel on the servers' devices.
    Train {
        /// Server to draw devices from: `ws://host:port[#i,j]` or `iroh://topic#i,j`.
        /// Repeat the flag for several servers.
        #[arg(long = "server", required = true)]
        servers: Vec<ServerDevices>,

        #[arg(long, value_enum, default_value_t = Strategy::Multi)]
        strategy: Strategy,

        #[arg(long, default_value_t = 256)]
        batch_size: usize,

        #[arg(long, default_value_t = 5)]
        epochs: usize,

        /// Dataloader workers per device.
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Strategy {
    /// One device does everything.
    Single,
    /// Data-parallel: gradients gathered and optimized on the first device.
    Multi,
    /// Data-parallel: optimizer state sharded across the devices.
    MultiSharded,
    /// Distributed data-parallel: gradients synced by all-reduce on the server. Needs every
    /// device on one server whose backend implements collectives (CUDA today).
    Ddp,
}

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();

    match Cli::parse().command {
        Command::Server { listen } => server::serve(&listen),
        Command::Train {
            servers,
            strategy,
            batch_size,
            epochs,
            workers,
        } => train(servers, strategy, batch_size, epochs, workers),
    }
}

fn train(
    servers: Vec<ServerDevices>,
    strategy: Strategy,
    batch_size: usize,
    epochs: usize,
    workers: usize,
) {
    if matches!(strategy, Strategy::Ddp) && servers.len() > 1 {
        eprintln!(
            "DDP needs every device on one server: cross-server all-reduce is not supported. \
             Use `--strategy multi` across servers, or a single `--server`."
        );
        std::process::exit(1);
    }

    // The endpoint's tasks live on this runtime, so it must outlive the training run.
    let runtime = tokio::runtime::Runtime::new().expect("Failed to start a tokio runtime");
    let endpoint = servers.iter().any(ServerDevices::needs_endpoint).then(|| {
        runtime
            .block_on(Endpoint::builder(presets::N0).bind())
            .expect("Failed to bind an iroh endpoint")
    });

    let devices: Vec<Device> = servers
        .iter()
        .flat_map(|spec| spec.connect(endpoint.as_ref()))
        .collect();
    println!("training on {} remote device(s)", devices.len());

    let strategy = execution_strategy(strategy, devices);
    training::run(
        strategy,
        TrainConfig {
            batch_size,
            num_epochs: epochs,
            num_workers: workers,
        },
    );
}

fn execution_strategy(strategy: Strategy, devices: Vec<Device>) -> ExecutionStrategy {
    match strategy {
        Strategy::Single => ExecutionStrategy::SingleDevice(
            devices
                .into_iter()
                .next()
                .expect("At least one server is required, and a server hosts at least one device"),
        ),
        Strategy::Multi => {
            ExecutionStrategy::MultiDevice(devices, MultiDeviceOptim::OptimMainDevice)
        }
        Strategy::MultiSharded => {
            ExecutionStrategy::MultiDevice(devices, MultiDeviceOptim::OptimSharded)
        }
        Strategy::Ddp => ExecutionStrategy::ddp(
            devices,
            DistributedConfig {
                all_reduce_op: ReduceOperation::Mean,
            },
        ),
    }
}
