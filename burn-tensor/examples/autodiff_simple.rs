use burn_tensor::tensor::{
    backend::autodiff::ADBackendNdArray,
    backend::ndarray::{NdArrayBackend, NdArrayTensor},
    backend::Backend,
    Data, Distribution, Shape, Tensor,
};

fn lossa<B: Backend>(x: &Tensor<2, B>, y: &Tensor<2, B>) -> Tensor<2, B> {
    let z = x.matmul(y);
    z
}

fn run() {
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    let x = Tensor::new(NdArrayTensor::from_data(x));
    let y = Tensor::new(NdArrayTensor::from_data(y));

    let z = lossa::<NdArrayBackend<f32>>(&x, &y);
    println!("Without AD");
    println!("z={}", z.to_data());

    let x = x.with_grad();
    let y = y.with_grad();

    let z = lossa::<ADBackendNdArray<f32>>(&x, &y);
    let grads = z.backward();

    println!("With AD");
    println!("z={}", z.to_data());
    println!("x_grad {}", x.grad(&grads).unwrap().to_data());
    println!("y_grad {}", y.grad(&grads).unwrap().to_data());
}

fn main() {
    run()
}
