use burn_compute::client::ComputeClient;
use burn_tensor::Shape;

use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        prng::base::{make_args_buffer, make_info_buffer},
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::JitTensor,
    IntElement, Runtime,
};

use super::base::Prng;

struct UniformPrng;
struct UniformIntPrng;

impl StaticKernelSource for UniformPrng {
    fn source() -> SourceTemplate {
        Prng::source().register("num_args", "2").register(
            "prng_loop",
            include_str!("../../template/prng/uniform_inner_loop.wgsl"),
        )
    }
}

impl StaticKernelSource for UniformIntPrng {
    fn source() -> SourceTemplate {
        Prng::source().register("num_args", "2").register(
            "prng_loop",
            include_str!("../../template/prng/uniform_int_inner_loop.wgsl"),
        )
    }
}

/// Pseudo-random generator for the uniform distribution.
pub fn random_uniform<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    low: E,
    high: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    uniform_kernel(client, device, &shape, low, high)
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: Runtime, E: JitElement, const D: usize>(
    tensor: &JitTensor<R, E, D>,
    low: E,
    high: E,
) -> JitTensor<R, E, D> {
    uniform_kernel(
        tensor.client.clone(),
        &tensor.device,
        &tensor.shape,
        low,
        high,
    )
}

/// Pseudo-random generator for uniform int distribution, based on
/// another tensor's client, device and shape.
pub fn random_like_uniform_int<R: Runtime, E: IntElement, const D: usize>(
    tensor: &JitTensor<R, E, D>,
    low: E,
    high: E,
) -> JitTensor<R, E, D> {
    uniform_int_kernel(
        tensor.client.clone(),
        &tensor.device,
        &tensor.shape,
        low,
        high,
    )
}

fn uniform_kernel<R: Runtime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: &R::Device,
    shape: &Shape<D>,
    low: E,
    high: E,
) -> JitTensor<R, E, D> {
    const N_VALUES_PER_THREAD: usize = 128;

    let output = empty_device(client.clone(), device.clone(), shape.clone());
    let info_handle = make_info_buffer::<R>(client.clone(), N_VALUES_PER_THREAD);
    let args_handle = make_args_buffer::<R, E>(client.clone(), &[low, high]);
    let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP_DEFAULT, N_VALUES_PER_THREAD);
    let kernel = StaticKernel::<
        KernelSettings<UniformPrng, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(workgroup);

    client.execute(
        Box::new(kernel),
        &[&output.handle, &info_handle, &args_handle],
    );

    output
}

fn uniform_int_kernel<R: Runtime, E: IntElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: &R::Device,
    shape: &Shape<D>,
    low: E,
    high: E,
) -> JitTensor<R, E, D> {
    const N_VALUES_PER_THREAD: usize = 128;

    let output = empty_device(client.clone(), device.clone(), shape.clone());
    let info_handle = make_info_buffer::<R>(client.clone(), N_VALUES_PER_THREAD);
    let args_handle = make_args_buffer::<R, E>(client.clone(), &[low, high]);
    let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP_DEFAULT, N_VALUES_PER_THREAD);
    let kernel = StaticKernel::<
        KernelSettings<UniformIntPrng, u32, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(workgroup);

    client.execute(
        Box::new(kernel),
        &[&output.handle, &info_handle, &args_handle],
    );

    output
}
