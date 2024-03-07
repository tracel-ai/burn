use burn_tensor::Shape;

use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};

use super::base::Prng;

struct NormalPrng;

/// Pseudo-random generaJitBackendl distribution
pub fn random_normal<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    mean: E,
    std: E,
) -> JitTensor<R, E, D> {
    todo!()
    // const N_VALUES_PER_THREAD: usize = 128; // must be even

    // let client = R::client(device);
    // let output = empty_device(client.clone(), device.clone(), shape.clone());
    // let info_handle = make_info_buffer::<R>(client.clone(), N_VALUES_PER_THREAD);
    // let args_handle = make_args_buffer::<R, E>(client.clone(), &[mean, std]);
    // let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP_DEFAULT, N_VALUES_PER_THREAD);
    // let kernel = StaticKernel::<
    //     KernelSettings<NormalPrng, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    // >::new(workgroup);

    // client.execute(
    //     Box::new(kernel),
    //     &[&output.handle, &info_handle, &args_handle],
    // );

    // output
}
