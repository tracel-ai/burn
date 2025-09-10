use std::time::Instant;

use burn::{
    collective::{self, CollectiveConfig, PeerId, ReduceOperation},
    prelude::*,
    tensor::TensorPrimitive,
};

pub fn run<B: Backend>(devices: Vec<B::Device>) {
    for strategy in [
        collective::AllReduceStrategy::Centralized,
        collective::AllReduceStrategy::Ring,
        collective::AllReduceStrategy::Tree(2),
    ] {
        println!("[All Reduce - {strategy:?}] starting ...");
        let start = Instant::now();
        task_all_reduce::<B>(
            devices.clone(),
            32,
            collective::AllReduceStrategy::Centralized,
        );
        println!("[All Reduce - {strategy:?}] took {:?}", start.elapsed());
    }
    task_different_tasks::<B>(devices.clone(), 100);
}

fn task_different_tasks<B: Backend>(mut devices: Vec<B::Device>, num_iterations: usize) {
    let aggregation_device = devices.pop().unwrap();

    let shape = [8, 4096, 4096];

    let (sender, receiver) = std::sync::mpsc::sync_channel(devices.len());

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
    let shape = [8, 4096, 4096];

    let handles = devices
        .into_iter()
        .enumerate()
        .map(|(id, device)| {
            std::thread::spawn(move || {
                let mut input =
                    Tensor::<B, 3>::random(shape, burn::tensor::Distribution::Default, &device);

                let id = PeerId::from(id);
                let config = CollectiveConfig::default()
                    .with_num_devices(num_devices)
                    .with_local_all_reduce_strategy(strategy);
                collective::register::<B>(id, device, config).unwrap();

                for i in 0..num_iterations {
                    let tensor = compute(input);
                    let result = collective::all_reduce::<B>(
                        id,
                        tensor.into_primitive().tensor(),
                        ReduceOperation::Mean,
                    )
                    .unwrap();
                    input = Tensor::from_primitive(TensorPrimitive::Float(result));
                    println!("[{id}] => Iter {i}");
                }
                collective::finish_collective::<B>(id).unwrap();
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        handle.join().unwrap();
    }
}

fn compute<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let log = input.clone() + 1.0;
    input.matmul(log)
}
