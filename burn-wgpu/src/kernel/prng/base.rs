use burn_common::rand::get_seeded_rng;
use burn_tensor::{Distribution, Shape};
use rand::Rng;

use crate::{
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice, SEED,
};

kernel_wgsl!(PRNG, "../../template/prng/default.wgsl");

pub fn random<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    distribution: Distribution<E>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let mut rng = match SEED.lock().unwrap().as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };
    let seed: u32 = rng.gen();

    let context = get_context::<G>(device);
    let num_elems = shape.num_elements();
    let buffer = context.create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(context.clone(), shape, buffer);

    let kernel = context.compile_static::<KernelSettings<PRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    println!("{:?}", seed);    
    let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&vec![seed]));

    context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&output.buffer, &info_buffer],
    );

    // For now, only default distribution (uniform in [0,1[)
    // Then:
    // match distribution {
    //     Distribution::Default => keep,
    //     Distribution::Bernoulli(prob) => greater,
    //     Distribution::Uniform(from, to) => stretch,
    //     Distribution::Normal(mean, std) => box-muller
    // }
    // transform default to distribution
    // replace by each kernel later

    // ?
    // *seed = Some(rng);

    output
}



#[cfg(test)]
mod tests {
    use burn_tensor::{Distribution, Tensor};

    use crate::{tests::TestBackend, WgpuDevice};

    #[test]
    fn print_out() {
        let shape = [2, 3];
        let device = WgpuDevice::default(); 
        let tensor = Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);
        println!("{:?}", tensor.into_data());
        assert!(false);
    }
    
}