use burn_tensor::tensor::{
    backend::autodiff::ADBackendNdArray,
    backend::autodiff::ADTensor,
    backend::ndarray::{NdArrayBackend, NdArrayTensor},
    backend::Backend,
    ops::*,
    Data, Distribution, Shape, Tensor,
};

fn loss<B: Backend>(x: &Tensor<2, B>, y: &Tensor<2, B>) -> Tensor<2, B> {
    let z = x.matmul(y);
    z
}

fn run() {
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    let x = Tensor::<2, NdArrayBackend<f32>>::new(NdArrayTensor::from_data(x));
    let y = Tensor::<2, NdArrayBackend<f32>>::new(NdArrayTensor::from_data(y));

    let z = loss(&x, &y);

    println!("without AD {}", z.to_data());
    let x = Tensor::<2, ADBackendNdArray<f32>>::new(ADTensor::from_tensor(x.value));
    let y = Tensor::<2, ADBackendNdArray<f32>>::new(ADTensor::from_tensor(y.value));

    let z = loss(&x, &y);
    let grads = z.value.backward();

    println!("X {}", grads.wrt(&x.value).unwrap().to_data());
    println!("Y {}", grads.wrt(&y.value).unwrap().to_data());
}

fn main() {
    run()
}
