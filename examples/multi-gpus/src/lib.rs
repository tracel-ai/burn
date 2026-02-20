use burn::{
    self,
    backend::Autodiff,
    collective::{CollectiveConfig, all_reduce, finish_collective, register},
    data::{dataloader::DataLoaderBuilder, dataset::transform::PartialDataset},
    nn::transformer::TransformerEncoderConfig,
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::{
        TensorPrimitive,
        backend::{AllReduceStrategy, AutodiffBackend, DeviceId, PeerId, ReduceOperation},
    },
};
use std::{
    sync::{Arc, mpsc::SyncSender},
    time::Instant,
};
use text_classification::{
    AgNewsDataset, TextClassificationDataset,
    data::{TextClassificationBatcher, Tokenizer},
    model::TextClassificationModel,
};

pub fn run<B: Backend>() {
    let type_id = 0;
    let num_devices = B::Device::device_count(type_id);

    let devices = (0..num_devices)
        .map(|i| B::Device::from_id(DeviceId::new(type_id, i as u32)))
        .collect();

    run_with::<B>(devices);
}

fn run_with<B: Backend>(devices: Vec<B::Device>) {
    for strategy in [
        AllReduceStrategy::Tree(2),
        AllReduceStrategy::Ring,
        AllReduceStrategy::Centralized,
    ] {
        println!("[Gradient Update - {strategy:?}] starting ...");
        let start = Instant::now();
        task_grad_all_reduce::<Autodiff<B>>(devices.clone(), strategy);
        println!(
            "[Gradient Update - {strategy:?}] took {:?}",
            start.elapsed()
        );
    }
    for strategy in [
        AllReduceStrategy::Centralized,
        AllReduceStrategy::Ring,
        AllReduceStrategy::Tree(2),
    ] {
        println!("[All Reduce - {strategy:?}] starting ...");
        let start = Instant::now();
        task_all_reduce::<B>(devices.clone(), 420, strategy);
        println!("[All Reduce - {strategy:?}] took {:?}", start.elapsed());
    }
    task_naive_aggregation::<B>(devices.clone(), 100);
}

fn task_naive_aggregation<B: Backend>(mut devices: Vec<B::Device>, num_iterations: usize) {
    let aggregation_device = devices.pop().unwrap();

    let shape = [8, 4096, 4096];

    let (sender, receiver) = std::sync::mpsc::sync_channel(devices.len());

    fn compute<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
        let log = input.clone() + 1.0;
        input.matmul(log)
    }

    let mut handles = devices
        .into_iter()
        .map(|device| {
            let sender = sender.clone();
            std::thread::spawn(move || {
                let input =
                    Tensor::<B, 3>::random(shape, burn::tensor::Distribution::Default, &device);

                for _ in 0..num_iterations {
                    let new = compute(input.clone());
                    sender.send(new.clone()).unwrap();
                }
            })
        })
        .collect::<Vec<_>>();

    handles.push(std::thread::spawn(move || {
        let mut input = Tensor::<B, 3>::random(
            shape,
            burn::tensor::Distribution::Default,
            &aggregation_device,
        );

        while let Ok(tensor) = receiver.recv() {
            let main = tensor.to_device(&aggregation_device);
            let value = main.clone().sum().into_scalar().elem::<f32>();
            input = input + main / 2;
            println!("{value:?}");
            assert_ne!(value, 0.0);
        }
    }));

    for handle in handles {
        handle.join().unwrap();
    }
}

fn task_all_reduce<B: Backend>(
    devices: Vec<B::Device>,
    num_iterations: usize,
    strategy: AllReduceStrategy,
) {
    let num_devices = devices.len();
    let batch = 32;
    let shape_signal = [batch, 2048, 2048];
    let shape_weights = [1, 2048, 2048];

    fn compute<B: Backend>(weights: Tensor<B, 3>, signal: Tensor<B, 3>) -> Tensor<B, 3> {
        weights.matmul(signal)
    }

    let handles = devices
        .into_iter()
        .enumerate()
        .map(|(id, device)| {
            std::thread::spawn(move || {
                let mut weights = Tensor::<B, 3>::random(
                    shape_weights,
                    burn::tensor::Distribution::Default,
                    &device,
                ) - 0.5;

                let id = PeerId::from(id);
                let config = CollectiveConfig::default()
                    .with_num_devices(num_devices)
                    .with_local_all_reduce_strategy(strategy);

                register::<B>(id, device.clone(), config).unwrap();

                for i in 0..num_iterations {
                    let signal = Tensor::<B, 3>::random(
                        shape_signal,
                        burn::tensor::Distribution::Default,
                        &device,
                    ) - 0.5;
                    let signal = compute(weights, signal);
                    let weights_update = signal.mean_dim(0);

                    let result = all_reduce::<B>(
                        id,
                        weights_update.into_primitive().tensor(),
                        ReduceOperation::Mean,
                    )
                    .unwrap();
                    weights = Tensor::from_primitive(TensorPrimitive::Float(result));
                    let val = weights.clone().sum().into_scalar().elem::<f32>();
                    if id == PeerId::from(0) {
                        println!("Iter {i} => {val}");
                    }
                }
                finish_collective::<B>(id).unwrap();
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}

fn task_grad_all_reduce<B: AutodiffBackend>(devices: Vec<B::Device>, strategy: AllReduceStrategy) {
    let num_devices = devices.len();
    let seq_length = nn::attention::SeqLengthOption::Fixed(512);
    let batch_size = 32;
    let config = TransformerEncoderConfig::new(256, 1024, 8, 4);

    let dataset = text_classification::AgNewsDataset::train();
    let tokenizer = Arc::new(text_classification::data::BertCasedTokenizer::default());
    let model_config = text_classification::model::TextClassificationModelConfig::new(
        config,
        AgNewsDataset::num_classes(),
        tokenizer.vocab_size(),
        seq_length,
    );
    let datasets = PartialDataset::split(dataset, devices.len());
    let model_main = model_config.init(&devices[0]);

    let handles = devices
        .into_iter()
        .zip(datasets)
        .enumerate()
        .map(|(id, (device, dataset))| {
            let model_main = model_main.clone();
            let tokenizer = tokenizer.clone();

            std::thread::spawn(move || {
                println!("[{id}] Running on device {device:?}");
                let mut model = model_main.fork(&device);
                let batcher = TextClassificationBatcher::new(tokenizer, seq_length);
                let dataloader_train = DataLoaderBuilder::new(batcher)
                    .batch_size(batch_size)
                    .set_device(device.clone())
                    .build(dataset);

                let syncher = GradSyncer::start::<B>(
                    CollectiveConfig::default()
                        .with_num_devices(num_devices)
                        .with_local_all_reduce_strategy(strategy),
                    device.clone(),
                    PeerId::from(id),
                );

                let mut optim = SgdConfig::new().init::<B, TextClassificationModel<B>>();

                for (i, batch) in dataloader_train.iter().enumerate() {
                    let output = model.forward(batch);
                    let loss: Tensor<B, 1> = output.loss.clone();

                    let grads = loss.backward();
                    let loss = loss.into_scalar().elem::<f32>();

                    let grads = GradientsParams::from_grads(grads, &model);
                    let grads = syncher.sync(grads);

                    if let Some(grads) = grads {
                        model = optim.step(1.0e-5, model, grads);
                    }

                    println!("[{id}] Iter {i} => {loss}");
                }
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}

struct GradSyncer {
    sender: SyncSender<Message>,
}

struct Message {
    callback: SyncSender<Option<GradientsParams>>,
    grads: GradientsParams,
}

impl GradSyncer {
    fn start<B: AutodiffBackend>(config: CollectiveConfig, device: Device<B>, id: PeerId) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Message>(8);

        std::thread::spawn(move || {
            println!("[{id}] Register peration {config:?}");
            register::<B::InnerBackend>(id, device, config).unwrap();
            let num_stages = 4;
            let mut buffers: Vec<GradientsParams> = Vec::new();

            while let Ok(msg) = receiver.recv() {
                let grads = msg
                    .grads
                    .all_reduce::<B::InnerBackend>(id, ReduceOperation::Mean)
                    .unwrap();

                buffers.push(grads);

                let result = if buffers.len() >= num_stages {
                    Some(buffers.remove(0))
                } else {
                    None
                };

                msg.callback.send(result).unwrap();
            }
            finish_collective::<B::InnerBackend>(id).unwrap();
        });

        Self { sender }
    }

    fn sync(&self, grads: GradientsParams) -> Option<GradientsParams> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let msg = Message {
            callback: sender,
            grads,
        };
        self.sender.send(msg).unwrap();

        receiver.recv().unwrap()
    }
}
