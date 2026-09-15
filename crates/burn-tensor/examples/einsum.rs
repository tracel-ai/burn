//! Run with `cargo run -p burn-tensor --example einsum --features flex,autodiff`.
use burn_tensor::{Device, Tensor, einsum};

fn main() {
    let device = Device::default();
    let a = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    let b = Tensor::<2>::from_floats([[5., 6.], [7., 8.]], &device);
    let product = einsum!("ij,jk->ik", &a, &b);
    println!(
        "Matrix multiplication: {:?}",
        product.into_data().try_to_vec::<f32>().unwrap()
    );

    let equation = String::from("ij,j->i");
    let vector = Tensor::<1>::from_floats([10., 20.], &device);
    let result = Tensor::<1>::einsum(&equation, [(&a).into(), vector.into()]);
    println!(
        "Runtime equation: {:?}",
        result.into_data().try_to_vec::<f32>().unwrap()
    );
    let trace = einsum!("ii->", &a);
    println!(
        "Trace: {:?}",
        trace.into_data().try_to_vec::<f32>().unwrap()
    );

    // MaskFormer: contract query channels with spatial feature channels.
    let queries = Tensor::<3>::from_floats([[[1., 2.], [3., 4.]]], &device);
    let features = Tensor::<4>::from_floats([[[[1., 2.]], [[3., 4.]]]], &device);
    let masks = einsum!("bqc,bchw->bqhw", queries, features);
    println!(
        "Mask prediction {:?}: {:?}",
        masks.dims(),
        masks.into_data().try_to_vec::<f32>().unwrap()
    );

    #[cfg(feature = "autodiff")]
    {
        let device = device.autodiff();
        let x = Tensor::<1>::from_floats([1., 2., 3.], &device).require_grad();
        let loss = einsum!("i,i->", &x, &x);
        let grads = loss.backward();
        println!(
            "Gradient of dot(x, x): {:?}",
            x.grad(&grads)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        );
    }
}
