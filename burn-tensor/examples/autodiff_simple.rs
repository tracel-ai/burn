use burn_tensor::tensor::backend::ndarray::NdArrayBackend;
use burn_tensor::tensor::backend::tch::{TchBackend, TchTensor};
use burn_tensor::tensor::{backend::autodiff::ADTensor, Backend, Tensor};
use burn_tensor::tensor::{
    ops::*, ADBackend2, Backend2, Data, Distribution, Shape, TchBackend2, Tensor2, TensorOps,
};
use std::time::{Duration, SystemTime};

fn loss<B: Backend2>(x: &Tensor2<2, B>, y: &Tensor2<2, B>) -> Tensor2<1, B>
where
    TensorOps<2, B>: TensorOpsReshape<B::Elem, 2, 1, TensorOps<1, B>>,
{
    let z = x.matmul(y);
    z.reshape(Shape::new([4]))
}

fn my_func<BForward: Backend>(
    x: Tensor<2, BForward>,
    y: Tensor<2, BForward>,
) -> (Data<BForward::E, 2>, Data<BForward::E, 2>, Duration) {
    let start = SystemTime::now();

    let x = ADTensor::from_tensor(x);
    let y = ADTensor::from_tensor(y);

    let loss = x.matmul(&y);

    let grads = loss.backward();

    let x_grad = grads.wrt(&x).expect("x gradient defined");
    let y_grad = grads.wrt(&y).expect("y gradient defined");

    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();

    (x_grad.to_data(), y_grad.to_data(), duration)
}

fn run<B: Backend>(x: Data<B::E, 2>, y: Data<B::E, 2>, device: B::Device) {
    let (x_grad, y_grad, duration) = my_func::<B>(B::from_data(x, device), B::from_data(y, device));

    println!("--- {} ---", B::name());
    println!("took: {} ns", duration.as_nanos());
    println!("x_grad: {}", x_grad);
    println!("y_grad: {}", y_grad);
}

fn trya<B: Backend2>(x: &Tensor2<2, B>)
where
    TensorOps<2, B>: TensorOpsIndex<B::Elem, 2, 3>,
    TensorOps<2, ADBackend2<B>>: TensorOpsReshape<B::Elem, 2, 1, TensorOps<1, ADBackend2<B>>>,
{
    x.add(&x);
    let x = x.track_grad();

    let loss = loss::<ADBackend2<B>>(&x, &x);
    println!("{}", loss.to_data());
}

fn main() {
    let x = Data::<f32, 2>::random(Shape::new([2, 2]), Distribution::Standard);

    let tensor = TchTensor::from_data(x, tch::Device::Cpu);
    let tensor: Tensor2<2, TchBackend2<f32>> = Tensor2::new(tensor);
    trya(&tensor)
}

fn main2() {
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    println!("x: {}", x);
    println!("y: {}", y);

    run::<NdArrayBackend<f32>>(
        x.clone(),
        y.clone(),
        burn_tensor::tensor::backend::ndarray::Device::Cpu,
    );
    run::<TchBackend<f32>>(
        x.clone(),
        y.clone(),
        burn_tensor::tensor::backend::tch::Device::Cpu,
    );
}
