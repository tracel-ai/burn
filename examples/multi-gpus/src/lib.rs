use burn::{
    backend::Autodiff,
    collective::{self, CollectiveConfig, PeerId, ReduceOperation},
    data::{dataloader::DataLoaderBuilder, dataset::transform::PartialDataset},
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig, decay::WeightDecayConfig},
    prelude::*,
    tensor::{
        TensorPrimitive,
        backend::{AutodiffBackend, DeviceId},
    },
};
use std::{sync::Arc, time::Instant};
use text_classification::{
    AgNewsDataset, TextClassificationDataset,
    data::{TextClassificationBatcher, Tokenizer},
    model::TextClassificationModel,
    training::ExperimentConfig,
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
        collective::AllReduceStrategy::Centralized,
        collective::AllReduceStrategy::Ring,
        collective::AllReduceStrategy::Tree(2),
    ] {
        println!("[Gradient Update - {strategy:?}] starting ...");
        let start = Instant::now();
        task_grad_all_reduce::<Autodiff<B>>(devices.clone(), 32, strategy);
        println!(
            "[Gradient Update - {strategy:?}] took {:?}",
            start.elapsed()
        );
    }
    for strategy in [
        collective::AllReduceStrategy::Centralized,
        collective::AllReduceStrategy::Ring,
        collective::AllReduceStrategy::Tree(2),
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
    strategy: collective::AllReduceStrategy,
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

                collective::register::<B>(id, device.clone(), config).unwrap();

                for i in 0..num_iterations {
                    let signal = Tensor::<B, 3>::random(
                        shape_signal,
                        burn::tensor::Distribution::Default,
                        &device,
                    ) - 0.5;
                    let signal = compute(weights, signal);
                    let weights_update = signal.mean_dim(0);

                    let result = collective::all_reduce::<B>(
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
                collective::finish_collective::<B>(id).unwrap();
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}

fn task_grad_all_reduce<B: AutodiffBackend>(
    devices: Vec<B::Device>,
    num_iterations: usize,
    strategy: collective::AllReduceStrategy,
) {
    let num_devices = devices.len();
    let seq_length = 256;
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    let dataset = text_classification::AgNewsDataset::train();
    let tokenizer = Arc::new(text_classification::data::BertCasedTokenizer::default());
    let model_config = text_classification::model::TextClassificationModelConfig::new(
        config.transformer,
        AgNewsDataset::num_classes(),
        tokenizer.vocab_size(),
        seq_length,
    );
    let datasets = PartialDataset::split(dataset, devices.len());

    let handles = devices
        .into_iter()
        .zip(datasets.into_iter())
        .enumerate()
        .map(|(id, (device, dataset))| {
            // let model_main = model_main.clone();
            let tokenizer = tokenizer.clone();
            let model_config = model_config.clone();

            std::thread::spawn(move || {
                let model = model_config.init(&device);
                // let mut model = model_main.fork(&device);
                let id = PeerId::from(id);
                let config_col = CollectiveConfig::default()
                    .with_num_devices(num_devices)
                    .with_local_all_reduce_strategy(strategy);
                let batcher = TextClassificationBatcher::new(tokenizer, seq_length);
                let dataloader_train = DataLoaderBuilder::new(batcher.clone())
                    .batch_size(config.batch_size)
                    .set_device(device.clone())
                    .build(dataset);

                // println!("[{id}] Register collective operation {config_col:?}");
                // collective::register::<B::InnerBackend>(id, device.clone(), config_col).unwrap();

                let mut optim = SgdConfig::new().init::<B, TextClassificationModel<B>>();

                for (i, batch) in dataloader_train.iter().enumerate() {
                    let output = model.forward(batch);
                    let loss: Tensor<B, 1> = output.loss.clone();

                    let grads = loss.backward();
                    // let stat = loss.into_scalar().elem::<f32>();

                    let grads = GradientsParams::from_grads(grads, &model);
                    // let grads = grads
                    //     .all_reduce::<B::InnerBackend>(id, ReduceOperation::Mean)
                    //     .unwrap();

                    model = optim.step(1.0e-5, model, grads);

                    if id == PeerId::from(0) {
                        println!("Iter {i}");
                        // println!("Iter {i} => {stat}");
                    }
                }
                // collective::finish_collective::<B::InnerBackend>(id).unwrap();
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}
