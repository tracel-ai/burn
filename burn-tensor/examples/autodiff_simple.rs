use burn_tensor::tensor::backend::ndarray::{NdArrayBackend, NdArrayTensor};
use burn_tensor::tensor::backend::tch::{TchBackend, TchTensor};
use burn_tensor::tensor::backend::TchDevice;
use burn_tensor::tensor::{backend::autodiff::ADTensor, Backend, Tensor};
use burn_tensor::tensor::{ops::*, Data, Distribution, Shape};
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
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    let (x_grad_ndarray, y_grad_ndarray, duration_ndarray) = my_func::<NdArrayBackend<f32>>(
        NdArrayTensor::from_data(x.clone()),
        NdArrayTensor::from_data(y.clone()),
    );

    let device = TchDevice::Cpu;
    let (x_grad_tch, y_grad_tch, duration_tch) = my_func::<TchBackend<f32>>(
        TchTensor::from_data(x.clone(), device),
        TchTensor::from_data(y.clone(), device),
    );

    println!("x: {}", x);
    println!("y: {}", y);

    println!("--- ndarray ---");
    println!("took: {} ns", duration_ndarray.as_nanos());
    println!("x_grad: {}", x_grad_ndarray);
    println!("y_grad: {}", y_grad_ndarray);

    println!("--- tch ---");
    println!("took: {} ns", duration_tch.as_nanos());
    println!("x_grad: {}", x_grad_tch);
    println!("y_grad: {}", y_grad_tch);
}

fn main() {
    run()
}
