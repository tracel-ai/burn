use std::time::Instant;

use burn::{
    backend::Autodiff,
    collective::{self, CollectiveConfig, PeerId, ReduceOperation},
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::{
        TensorPrimitive,
        backend::{AutodiffBackend, DeviceId},
    },
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
    let batch = 32;
    let seq_length = 512;
    let d_model = 1024;
    let shape_signal = [batch, seq_length, d_model];
    let config = TransformerEncoderConfig::new(d_model, 2048, 4, 4);
    let model_main = config.init::<B>(&devices[0]);

    let handles = devices
        .into_iter()
        .enumerate()
        .map(|(id, device)| {
            let model_main = model_main.clone();

            std::thread::spawn(move || {
                let mut model = model_main.fork(&device);
                let id = PeerId::from(id);
                let config_col = CollectiveConfig::default()
                    .with_num_devices(num_devices)
                    .with_local_all_reduce_strategy(strategy);

                println!("[{id}] Register collective operation {config_col:?}");
                collective::register::<B::InnerBackend>(id, device.clone(), config_col).unwrap();

                let mut optim = SgdConfig::new().init::<B, TransformerEncoder<B>>();

                for i in 0..num_iterations {
                    let x = Tensor::<B, 3>::random(
                        shape_signal,
                        burn::tensor::Distribution::Default,
                        &device,
                    ) - 0.5;

                    let x = TransformerEncoderInput::new(x);
                    let x = model.forward(x);
                    let sum = x.sum();

                    let grads = sum.backward();
                    let stat = sum.into_scalar().elem::<f32>();

                    let grads = GradientsParams::from_grads(grads, &model);
                    let grads = grads
                        .all_reduce::<B::InnerBackend>(id, ReduceOperation::Mean)
                        .unwrap();

                    model = optim.step(1.0e-5, model, grads);

                    if id == PeerId::from(0) {
                        println!("Iter {i} => {stat}");
                    }
                }
                collective::finish_collective::<B::InnerBackend>(id).unwrap();
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}
