use burn::{
    random,
    tensor::{ops::*, ADTensor, Backend, Data, NdArrayTensorBackend, TchTensorGPUBackend, Tensor},
};
use burn_tensor::tensor::Distribution;
use std::time::{Duration, SystemTime};

fn my_func<B: Backend>(
    x: Tensor<2, B>,
    y: Tensor<2, B>,
) -> (Data<B::E, 2>, Data<B::E, 2>, Duration) {
    let start = SystemTime::now();

    let x = ADTensor::from_tensor(x);
    let y = ADTensor::from_tensor(y);

    let z = x.matmul(&y);

    let grads = z.backward();

    let x_grad = grads.wrt(&x).expect("x gradient defined");
    let y_grad = grads.wrt(&y).expect("y gradient defined");

    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();

    (x_grad.to_data(), y_grad.to_data(), duration)
}

fn run() {
    let x = random!(
        elem: f32,
        shape: [2, 3],
        distribution: Distribution::Standard,
        backend: ndarray
    );
    let y = random!(
        elem: f32,
        shape: [3, 1],
        distribution: Distribution::Standard,
        backend: ndarray
    );

    let (x_grad, y_grad, duration) = my_func::<NdArrayTensorBackend<f32>>(x, y);

    println!("--- ndarray ---");
    println!("took: {} ns", duration.as_nanos());
    println!("x_grad: {}", x_grad);
    println!("y_grad: {}", y_grad);

    let x = random!(
        elem: f32,
        shape: [2, 3],
        distribution: Distribution::Standard,
        backend: tch cpu
    );
    let y = random!(
        elem: f32,
        shape: [3, 1],
        distribution: Distribution::Standard,
        backend: tch cpu
    );

    let (x_grad, y_grad, duration) = my_func::<TchTensorGPUBackend<f32, 1>>(x, y);

    println!("--- tch ---");
    println!("took: {} ns", duration.as_nanos());
    println!("x_grad: {}", x_grad);
    println!("y_grad: {}", y_grad);
}

fn main() {
    run()
}
