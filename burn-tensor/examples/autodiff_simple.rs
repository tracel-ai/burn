use burn_tensor::{activation, backend, Data, Distribution, Shape, Tensor};
use rand::{rngs::StdRng, SeedableRng};

fn loss<B: backend::Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>) -> Tensor<B, 2> {
    let z = x.matmul(y);
    let z = activation::relu(&z);

    println!("fn name  : loss");
    println!("backend  : {}", B::name());
    println!("z        : {}", z.to_data());
    println!("autodiff : {}", B::ad_enabled());
    z
}

fn run_ad<B: backend::ADBackend>(x: Data<B::Elem, 2>, y: Data<B::Elem, 2>) {
    println!("---------- Ad Enabled -----------");
    let x: Tensor<B, 2> = Tensor::from_data(x);
    let y: Tensor<B, 2> = Tensor::from_data(y);

    let z = loss(&x, &y);

    let grads = z.backward();

    println!("x_grad   : {}", x.grad(&grads).unwrap().to_data());
    println!("y_grad   : {}", y.grad(&grads).unwrap().to_data());
    println!("---------------------------------");
    println!()
}

fn run<B: backend::Backend>(x: Data<B::Elem, 2>, y: Data<B::Elem, 2>) {
    println!("---------- Ad Disabled ----------");
    loss::<B>(&Tensor::from_data(x), &Tensor::from_data(y));
    println!("---------------------------------");
    println!()
}

fn main() {
    // Same data for all backends
    let mut rng = StdRng::from_entropy();
    let x = Data::random(Shape::new([2, 3]), Distribution::Standard, &mut rng);
    let y = Data::random(Shape::new([3, 1]), Distribution::Standard, &mut rng);

    #[cfg(feature = "ndarray")]
    {
        run::<backend::NdArrayBackend<f32>>(x.clone(), y.clone());
        run_ad::<backend::NdArrayADBackend<f32>>(x, y);
    }

    #[cfg(feature = "tch")]
    {
        run::<backend::TchBackend<f32>>(x.clone(), y.clone());
        run_ad::<backend::TchADBackend<f32>>(x, y);
    }
}
