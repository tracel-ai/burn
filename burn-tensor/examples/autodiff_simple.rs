use burn_tensor::{back, Data, Distribution, Shape, Tensor};

/// This function works for all backends
fn loss<B: back::Backend>(x: &Tensor<2, B>, y: &Tensor<2, B>) -> Tensor<2, B> {
    let z = x.matmul(y);
    z
}

/// This function requires a backend that can backward and compute gradients
fn run<B: back::ad::Backend>(x: Data<B::Elem, 2>, y: Data<B::Elem, 2>) {
    let x: Tensor<2, B> = Tensor::from_data(x);
    let y: Tensor<2, B> = Tensor::from_data(y);

    let z = loss(&x, &y);

    let grads = z.backward();

    println!("z={} with ad", z.to_data());
    println!("x_grad {}", x.grad(&grads).unwrap().to_data());
    println!("y_grad {}\n", y.grad(&grads).unwrap().to_data());
}

fn main() {
    // Same data for all backends
    let x = Data::<f32, 2>::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::<f32, 2>::random(Shape::new([3, 1]), Distribution::Standard);

    #[cfg(feature = "ndarray")]
    {
        println!("=== ndarray Backend ===\n");

        // NO AD
        let z = loss::<back::NdArray<f32>>(
            &Tensor::from_data(x.clone()),
            &Tensor::from_data(y.clone()),
        );

        // WITH AD
        println!("z={} without ad", z.to_data());
        run::<back::ad::NdArray<f32>>(x.clone(), y.clone());
    }

    #[cfg(feature = "tch")]
    {
        println!("=== Tch Backend ===\n");

        // NO AD
        let z =
            loss::<back::Tch<f32>>(&Tensor::from_data(x.clone()), &Tensor::from_data(y.clone()));

        // WITH AD
        println!("z={} without ad", z.to_data());
        run::<back::ad::Tch<f32>>(x.clone(), y.clone());
    }
}
