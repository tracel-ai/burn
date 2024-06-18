use burn::{
    backend::wgpu::WgpuRuntime,
    tensor::{Distribution, Tensor},
};
use custom_wgpu_kernel::{
    matmul_add_relu_custom, matmul_add_relu_reference, AutodiffBackend, Backend,
};

fn inference<B: Backend>(device: &B::Device) {
    let lhs = Tensor::<B, 3>::random([1, 32, 32], Distribution::Default, device);
    let rhs = Tensor::random([32, 32, 32], Distribution::Default, device);
    let bias = Tensor::random([32, 32, 32], Distribution::Default, device);

    let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone())
        .into_data()
        .convert::<f32>();
    let custom = matmul_add_relu_custom(lhs, rhs, bias)
        .into_data()
        .convert::<f32>();

    reference.assert_approx_eq(&custom, 3);

    println!("Both reference and the custom fused kernel have the same output");
}

fn autodiff<B: AutodiffBackend>(device: &B::Device) {
    let lhs = Tensor::<B, 3>::random([1, 32, 32], Distribution::Default, device).require_grad();
    let rhs = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();
    let bias = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();

    let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone());

    let mut gradients = reference.backward();

    let lhs_grad_ref = lhs.grad_remove(&mut gradients).unwrap();
    let rhs_grad_ref = rhs.grad_remove(&mut gradients).unwrap();
    let bias_grad_ref = bias.grad_remove(&mut gradients).unwrap();

    let lhs = lhs.detach();
    let rhs = rhs.detach();
    let bias = bias.detach();

    let custom = matmul_add_relu_custom(lhs.clone(), rhs.clone(), bias.clone());

    let mut gradients = custom.backward();

    let lhs_grad_custom = lhs.grad_remove(&mut gradients).unwrap();
    let rhs_grad_custom = rhs.grad_remove(&mut gradients).unwrap();
    let bias_grad_custom = bias.grad_remove(&mut gradients).unwrap();

    lhs_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&lhs_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same lhs gradient");

    rhs_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&rhs_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same rhs gradient");

    bias_grad_ref
        .into_data()
        .convert::<f32>()
        .assert_approx_eq(&bias_grad_custom.into_data().convert(), 3);

    println!("Both reference and the custom fused kernel have the same bias gradient");
}

fn main() {
    type MyBackend = burn::backend::wgpu::JitBackend<WgpuRuntime, f32, i32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
    let device = Default::default();
    inference::<MyBackend>(&device);
    autodiff::<MyAutodiffBackend>(&device);
}
