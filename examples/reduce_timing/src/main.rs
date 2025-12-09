#![allow(unused)]

use burn::Tensor;
use burn::collective::{
    AllReduceStrategy, CollectiveConfig, PeerId, ReduceOperation, all_reduce, register,
};
use burn::prelude::{Backend, Device, DeviceOps, TensorData, s};
use burn::tensor::backend::DeviceId;
use burn::tensor::{Distribution, Shape, TensorPrimitive};
use clap::{Parser, ValueEnum};
use opentelemetry::global::tracer;
use opentelemetry::trace::{Tracer, TracerProvider};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Sampler;
use std::env::args;
use std::error::Error;
use std::iter::repeat_with;
use std::num::ParseIntError;
use std::process::id;
use std::sync::mpsc::Receiver;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::instrument::WithSubscriber;
use tracing::{instrument, subscriber};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::try_init;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry, fmt};

fn parse_array4(s: &str) -> Result<[usize; 4], String> {
    let parts: Result<Vec<_>, _> = s.split(',').map(|p| p.trim().parse()).collect();
    let parts = parts.map_err(|e: ParseIntError| e.to_string())?;
    parts
        .try_into()
        .map_err(|v: Vec<_>| format!("expected 4 values, got {}", v.len()))
}

fn parse_all_reduce_strategy(s: &str) -> Result<AllReduceStrategy, String> {
    let s = s.trim();
    if s.starts_with("tree:") {
        let depth = s[5..].parse::<usize>().map_err(|e| e.to_string())?;
        Ok(AllReduceStrategy::Tree(depth as u32))
    } else if s.eq("centralized") {
        Ok(AllReduceStrategy::Centralized)
    } else if s.eq("ring") {
        Ok(AllReduceStrategy::Ring)
    } else {
        Err(format!("unknown strategy: {}", s))
    }
}

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

#[derive(Parser, Debug)]
pub struct Args {
    /// Supress verbose output.
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
}

impl Args {
    pub fn quiet(&self) -> bool {
        self.quiet.unwrap_or(false)
    }

    pub fn verbose(&self) -> bool {
        !self.quiet()
    }
}

#[tracing::instrument]
fn test_event() {
    tracing::info!("test event");

    let span = tracing::info_span!("test_span");
    let _guard = span.enter();
    tracing::info!("inside span");
}

static APP_NAME: &str = "reduce_timing";

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
            opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

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
                .with(telemetry)
                .with(env_filter)
                .try_init()?;

            Some(provider)
        }
    };

    test_event();

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda>(&args)?;

    if let Some(provider) = tracing_provider {
        if args.verbose() {
            println!("> main: shutting down tracing");
        }
        provider.shutdown();
    }

    Ok(())
}

struct Worker<B: Backend> {
    index: usize,
    id: PeerId,
    device: B::Device,
    config: CollectiveConfig,
}

impl<B: Backend> Worker<B> {
    pub fn new(index: usize, device: B::Device, config: CollectiveConfig) -> Self {
        let device = device.clone();
        let id = index.into();
        Self {
            index,
            id,
            device,
            config,
        }
    }

    #[tracing::instrument(skip(self, tensor))]
    pub fn all_reduce<const R: usize>(
        &mut self,
        tensor: Tensor<B, R>,
        op: ReduceOperation,
    ) -> Tensor<B, R> {
        eprintln!("w={}: all_reduce start", self.index);
        let tensor = Tensor::from_primitive(TensorPrimitive::Float(
            all_reduce::<B>(self.id, tensor.into_primitive().tensor(), op).unwrap(),
        ));
        eprintln!("w={}: all_reduce end", self.index);
        tensor
    }

    pub fn run(&mut self, rx: Receiver<WorkRequest<B>>) {
        println!("worker {} started", self.index);
        while let Ok(command) = rx.recv() {
            use WorkRequest::*;
            match command {
                RegisterRequest { tx } => {
                    register::<B>(self.id, self.device.clone(), self.config.clone()).unwrap();
                    tx.send(()).unwrap();
                }
                AllReduceRequest { tensor, op, tx } => {
                    assert_eq!(&tensor.device(), &self.device);
                    let tensor = self.all_reduce(tensor, op);
                    tx.send(tensor).unwrap();
                }
            }
        }
    }
}

pub enum WorkRequest<B: Backend> {
    RegisterRequest {
        tx: std::sync::mpsc::SyncSender<()>,
    },
    AllReduceRequest {
        tensor: Tensor<B, 4>,
        op: ReduceOperation,
        tx: std::sync::mpsc::SyncSender<Tensor<B, 4>>,
    },
}

struct WorkerHandle<B: Backend> {
    device: B::Device,
    tx: std::sync::mpsc::SyncSender<WorkRequest<B>>,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> WorkerHandle<B> {
    #[tracing::instrument(skip(config))]
    pub fn new(index: usize, device: &B::Device, config: CollectiveConfig) -> Self {
        let type_id = 0;
        let mut worker: Worker<B> = Worker::new(index, device.clone(), config.clone());

        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || worker.run(rx));
        Self {
            device: device.clone(),
            tx,
            phantom: Default::default(),
        }
    }

    pub fn register(&self) -> Receiver<()> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.tx.send(WorkRequest::RegisterRequest { tx }).unwrap();
        rx
    }

    pub fn to_device<const R: usize>(&self, tensor: Tensor<B, R>) -> Tensor<B, R> {
        tensor.to_device(&self.device)
    }

    pub fn all_reduce(&self, op: ReduceOperation, tensor: Tensor<B, 4>) -> Receiver<Tensor<B, 4>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(WorkRequest::AllReduceRequest { tensor, op, tx })
            .unwrap();
        rx
    }
}

#[allow(unused)]
#[tracing::instrument(skip(args))]
fn run<B: Backend>(args: &Args) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    let type_id = 0;
    let device_count = B::Device::device_count(type_id);

    let devices = (0..device_count)
        .into_iter()
        .map(|idx| B::Device::from_id(DeviceId::new(type_id, idx as u32)))
        .collect::<Vec<_>>();

    /// Duplicate the devices to force a heterogeneous setup.
    let devices = devices
        .iter()
        .flat_map(|x| repeat_with(|| x.clone()).take(args.workers_per_device))
        .collect::<Vec<_>>();

    let config = CollectiveConfig::default()
        .with_num_devices(devices.len())
        .with_local_all_reduce_strategy(args.strategy.clone());
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
        .map(|rx| rx.recv())
        .collect::<Result<(), _>>()?;

    let shape: Shape = args.shape.into();

    let expected_sum = handles.len() * (handles.len() + 1) / 2;
    let expected =
        Tensor::<B, 4>::full(shape.clone(), expected_sum as f32, &B::Device::default()).to_data();

    if args.verbose() {
        println!("> run: setting up device tensors: {:?}", shape);
    }
    let tensors = handles
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            let full_value = idx as f32 + 1.0;
            Tensor::<B, 4>::full(shape.clone(), full_value, &h.device)
        })
        .collect::<Vec<_>>();

    if args.verbose() {
        println!("> run: running all_reduce");
    }
    let reduced = handles
        .iter()
        .zip(tensors.into_iter())
        .map(|(h, t)| h.all_reduce(ReduceOperation::Sum, t))
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
