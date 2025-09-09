use burn::prelude::*;

pub fn run<B: Backend>(mut devices: Vec<B::Device>) {
    let aggregation_device = devices.pop().unwrap();

    let shape = [8, 4096, 4096];
    let num_iterations = 1000;

    let (sender, receiver) = std::sync::mpsc::sync_channel(32);

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
                    let _ = new.sum().into_scalar();
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
            println!("{tensor}");
            let main = tensor.to_device(&aggregation_device);
            input = input + main.clone() / 2;
            let value = main.sum().into_scalar().elem::<f32>();
            println!("{value:?}");
            assert_ne!(value, 0.0);
        }
    }));

    for handle in handles {
        handle.join().unwrap();
    }
}

fn compute<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    let log = input.clone() + 1.0;
    input.matmul(log)
}
