use burn::tensor::*;
use burn_tensor::backend::autodiff::ADTensor;

fn main() {
    let x = random!([2, 3]);
    let y = random!([3, 2]);

    println!("x: {}", x.to_data());
    println!("y: {}", y.to_data());

    let x = ADTensor::from_tensor(x);
    let y = ADTensor::from_tensor(y);

    let z = x.matmul(&y);

    let grads = z.backward();

    let x_grad = grads.wrt(&x).expect("x gradient defined");
    let y_grad = grads.wrt(&y).expect("y gradient defined");

    println!("x_grad: {}", x_grad.to_data());
    println!("y_grad: {}", y_grad.to_data());
}
