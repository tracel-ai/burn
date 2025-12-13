use crate::event_utils::example_instrumented_event;
use crate::workers::WorkerHandle;
use burn::Tensor;
use burn::collective::{AllReduceStrategy, CollectiveConfig, ReduceOperation};
use burn::prelude::{Backend, DeviceOps};
use burn::tensor::Shape;
use burn::tensor::backend::DeviceId;
use clap::{Parser, ValueEnum};
use opentelemetry::trace::TracerProvider;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use std::error::Error;
use std::iter::repeat_with;
use std::sync::mpsc::Receiver;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod parsers;
use parsers::*;
mod event_utils;
mod workers;

static APP_NAME: &str = "dop_timer";

#[derive(Debug, Clone, ValueEnum)]
pub enum ConsoleFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum TracingMode {
    /// Print to stderr.
    Console,

    /// Export to OTEL via gRPC.
    Otel,
}

/// Timing tool for measuring the performance of collective operations.
///
/// Currently only supports `all_reduce`.
#[derive(Parser, Debug)]
pub struct Args {
    /// Suppress verbose output.
    #[arg(long, action = clap::ArgAction::Set, default_value = None)]
    pub quiet: Option<bool>,

    /// Enable tracing.
    #[arg(long, value_enum, default_value = None)]
    pub tracing: Option<TracingMode>,

    /// Output format for console tracing.
    #[arg(long, value_enum, default_value = "text")]
    pub console_tracing_format: ConsoleFormat,

    // TODO: sub-commands.
    /// Shape of the tensor to reduce.
    #[arg(long, value_parser = parse_array4, default_value = "2, 3, 256, 256")]
    pub shape: [usize; 4],

    /// All-reduce strategy.
    #[arg(long, value_parser = parse_all_reduce_strategy, default_value = "tree:2")]
    pub strategy: AllReduceStrategy,

    /// Number of workers per device.
    #[arg(long, default_value = "1")]
    pub workers_per_device: usize,

    /// Reduce operation.
    #[arg(long, value_parser = parse_reduce_operation, default_value = "sum")]
    pub op: ReduceOperation,
}

impl Args {
    pub fn quiet(&self) -> bool {
        self.quiet.unwrap_or(false)
    }

    pub fn verbose(&self) -> bool {
        !self.quiet()
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let args = Args::parse();
    if args.verbose() {
        println!("{:?}", args);
    }

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let tracing_provider = match &args.tracing {
        None => None,
        Some(TracingMode::Console) => {
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT);

            let subscriber = subscriber.with_writer(std::io::stderr);

            match &args.console_tracing_format {
                ConsoleFormat::Text => subscriber.try_init()?,
                ConsoleFormat::Json => subscriber.json().try_init()?,
            };

            None
        }
        Some(TracingMode::Otel) => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .build()?;

            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_batch_exporter(exporter)
                .with_sampler(opentelemetry_sdk::trace::Sampler::AlwaysOn)
                .with_resource(Resource::builder().with_service_name(APP_NAME).build())
                .build();

            opentelemetry::global::set_tracer_provider(provider.clone());

            let tracer = provider.tracer(APP_NAME);

            let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

            tracing_subscriber::registry()
                .with(env_filter)
                .with(telemetry)
                .try_init()?;

            opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

            Some(provider)
        }
    };

    example_instrumented_event();

    let count = [
        cfg!(feature = "cuda"),
        cfg!(feature = "metal"),
        cfg!(feature = "wgpu"),
        cfg!(feature = "ndarray"),
    ]
    .iter()
    .filter(|x| **x)
    .count();
    assert_eq!(count, 1, "exactly one backend must be enabled");

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda>(&args)?;

    #[cfg(feature = "metal")]
    run::<burn::backend::Metal>(&args)?;

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(&args)?;

    #[cfg(feature = "ndarray")]
    run::<burn::backend::NdArray>(&args)?;

    if let Some(provider) = tracing_provider {
        if args.verbose() {
            println!("> main: shutting down tracing");
        }
        provider.shutdown()?;
    }

    Ok(())
}

#[allow(unused)]
#[tracing::instrument(skip(args))]
fn run<B: Backend>(args: &Args) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let type_id = 0;
    let device_count = B::Device::device_count(type_id);

    let devices = (0..device_count)
        .map(|idx| B::Device::from_id(DeviceId::new(type_id, idx as u32)))
        .collect::<Vec<_>>();

    /// Duplicate the devices to force a heterogeneous setup.
    let devices = devices
        .iter()
        .flat_map(|x| repeat_with(|| x.clone()).take(args.workers_per_device))
        .collect::<Vec<_>>();

    let config = CollectiveConfig::default()
        .with_num_devices(devices.len())
        .with_local_all_reduce_strategy(args.strategy);
    if args.verbose() {
        println!("> run: config:\n{:#?}", config);
    }

    let handles: Vec<WorkerHandle<B>> = devices
        .iter()
        .enumerate()
        .map(|(idx, device)| WorkerHandle::new(idx, device, config.clone()))
        .collect();

    if args.verbose() {
        println!("> run: registering workers");
    }
    handles
        .iter()
        .map(|h| h.register())
        // Introduce a sequence point (wait for all workers to start)
        .collect::<Vec<_>>()
        // Wait on all results.
        .into_iter()
        .try_for_each(|rx: Receiver<()>| rx.recv())?;

    let shape: Shape = args.shape.into();

    let expected_cell: f32 = {
        let count = handles.len() as f32;
        let sum = (count * (count + 1.0)) / 2.0;
        if args.op == ReduceOperation::Mean {
            sum / count
        } else {
            sum
        }
    };
    let expected =
        Tensor::<B, 4>::full(shape.clone(), expected_cell, &B::Device::default()).to_data();

    if args.verbose() {
        println!("> run: setting up device tensors: {:?}", shape);
    }
    let tensors = handles
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            let full_value = idx as f32 + 1.0;
            Tensor::<B, 4>::full(shape.clone(), full_value, h.device())
        })
        .collect::<Vec<_>>();

    if args.verbose() {
        println!("> run: running all_reduce");
    }
    let reduced: Vec<Tensor<B, 4>> = handles
        .iter()
        .zip(tensors.into_iter())
        .map(|(h, t)| h.all_reduce(args.op, t))
        // Introduce a sequence point.
        .collect::<Vec<_>>()
        // Wait on all results.
        .into_iter()
        .map(|rx| rx.recv())
        .collect::<Result<Vec<_>, _>>()?;

    if args.verbose() {
        println!("> run: verifying result");
    }
    for t in reduced {
        t.into_data().assert_eq(&expected, true);
    }

    Ok(())
}
