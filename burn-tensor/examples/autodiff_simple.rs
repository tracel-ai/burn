use burn_tensor::{af, back, Data, Distribution, Shape, Tensor};

fn loss<B: back::Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> Tensor<B, 2> {
    let z = x.matmul(y);
    let z = af::relu(&z);

    println!("fn name  : loss");
    println!("backend  : {}", B::name());
    println!("z        : {}", z.to_data());
    println!("autodiff : {}", B::ad_enabled());
    z
}

fn run_ad<B: back::ad::Backend>(x: Data<B::Elem, 2>, y: Data<B::Elem, 2>) {
    println!("---------- Ad Enabled -----------");
    let x: Tensor<B, 2> = Tensor::from_data(x);
    let y: Tensor<B, 2> = Tensor::from_data(y);

    let z = loss(&x, &y);

    let grads = z.backward();

    println!("x_grad   : {}", x.grad(&grads).unwrap().to_data());
    println!("y_grad   : {}", y.grad(&grads).unwrap().to_data());
    println!("---------------------------------");
    println!("")
}

fn run<B: back::Backend>(x: Data<B::Elem, 2>, y: Data<B::Elem, 2>) {
    println!("---------- Ad Disabled ----------");
    loss::<B>(&Tensor::from_data(x.clone()), &Tensor::from_data(y.clone()));
    println!("---------------------------------");
    println!("")
}

fn main() {
    // Same data for all backends
    let x = Data::random(Shape::new([2, 3]), Distribution::Standard);
    let y = Data::random(Shape::new([3, 1]), Distribution::Standard);

    #[cfg(feature = "ndarray")]
    {
        run::<back::NdArray<f32>>(x.clone(), y.clone());
        run_ad::<back::ad::NdArray<f32>>(x.clone(), y.clone());
    }

    #[cfg(feature = "tch")]
    {
        run::<back::Tch<f32>>(x.clone(), y.clone());
        run_ad::<back::ad::Tch<f32>>(x.clone(), y.clone());
    }
}
