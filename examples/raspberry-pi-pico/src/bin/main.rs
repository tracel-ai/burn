#![no_std]
#![no_main]

use burn::{backend::NdArray, tensor::Tensor};
use defmt::*;
use embassy_executor::Spawner;
use raspberry_pi_pico::sine::Model;
use {defmt_rtt as _, panic_probe as _};
use embassy_rp as _;
use embedded_alloc::Heap;

type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
    }

    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    // Define input
    let mut input = 0.0;
    loop {
        if input > 2.0 { input = 0.0 }
        input += 0.05;

        // Run the model
        let output = run_model(&model, &device, input);

        // Output the values
        match output.into_primitive().tensor().array.as_slice() {
            Some(slice) => info!("input: {} - output: {}", input, slice),
            None => defmt::panic!("Failed to get value")
        };
    }
}

fn run_model<'a>(model: &Model<NdArray>, device: &BackendDevice, input: f32) -> Tensor<Backend, 2> {
    // Define the tensor
    let input = Tensor::<Backend, 2>::from_floats([[input]], &device);

    // Run the model on the input
    let output = model.forward(input);

    output
}
