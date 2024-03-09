use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        prng::base::{make_args_buffer, make_info_buffer},
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT,
    },
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::Shape;

use super::base::Prng;

struct BernoulliPrng;

impl StaticKernelSource for BernoulliPrng {
    fn source() -> SourceTemplate {
        Prng::source()
            .register("num_args", "1")
            .register(
                "prng_loop",
                include_str!("../../template/prng/bernoulli_inner_loop.wgsl"),
            )
            .add_template("fn cast_elem(e: bool) -> {{ elem }} {return {{elem}}(e);}")
    }
}

/// Pseudo-random generator for bernoulli
pub fn random_bernoulli<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    prob: E,
) -> JitTensor<R, E, D> {
    const N_VALUES_PER_THREAD: usize = 128;

    let client = R::client(device);
    let output = empty_device(client.clone(), device.clone(), shape.clone());
    let info_handle = make_info_buffer::<R>(client.clone(), N_VALUES_PER_THREAD);
    let args_handle = make_args_buffer::<R, E>(client.clone(), &[prob]);
    let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP_DEFAULT, N_VALUES_PER_THREAD);
    let kernel = StaticKernel::<
        KernelSettings<BernoulliPrng, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(workgroup);

    client.execute(
        Box::new(kernel),
        &[&output.handle, &info_handle, &args_handle],
    );

    output
}
