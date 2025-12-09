#![no_std]
#![no_main]

use burn::{backend::NdArray, tensor::Tensor};
use embassy_executor::Spawner;
use embassy_rp::{
    bind_interrupts,
    gpio::{Level, Output},
    peripherals::USB,
    usb::{Driver, InterruptHandler},
};
use embassy_time::Timer;
use embedded_alloc::LlffHeap as Heap;
use raspberry_pi_pico::sine::Model;
use {defmt_rtt as _, panic_probe as _};

// Set the backend to NdArray with f32
type Backend = NdArray<f32>;
// Get the backend device we use (cpu)
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

#[global_allocator]
static HEAP: Heap = Heap::empty();

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

// This is the main function for the program. Where execution starts.
#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // Initializes the allocator, must be done before use.
    {
        use core::mem::MaybeUninit;
        // Watch out for this, if it is too big or small, program may crash
        // this is in u8 bytes, as such this is a total of 100kb
        const HEAP_SIZE: usize = 100 * 1024;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(&raw mut HEAP_MEM as usize, HEAP_SIZE) } // Initialize the heap
    }

    // This is just setup to make the microcontroller output to serial.
    let p = embassy_rp::init(Default::default());
    let driver = Driver::new(p.USB, Irqs);
    spawner.spawn(logger_task(driver)).unwrap();

    // Set the onboard LED to high to help indicate that the program is working properly.
    let mut led = Output::new(p.PIN_25, Level::Low);
    led.set_high();

    // Get a default device for the backend
    let device = BackendDevice::default();

    // Create a new model and load the state
    let model: Model<Backend> = Model::default();

    // Define input, this is the `x` in the function `y = sin(x)` that we are
    // approximating with our model
    let mut input = 0.0;
    loop {
        if input > 2.0 {
            input = 0.0
        }
        input += 0.05;

        // Run the model
        let output = run_model(&model, &device, input);

        // Output the values
        match output.into_data().as_slice::<f32>() {
            Ok(slice) => log::info!("input: {:.3} - output: {:?}", input, slice),
            Err(err) => core::panic!("err: {:?}", err),
        };

        Timer::after_millis(1).await; // Have some time to flush out the serial data
    }
}

fn run_model(model: &Model<NdArray>, device: &BackendDevice, input: f32) -> Tensor<Backend, 2> {
    // Define the tensor
    let input = Tensor::<Backend, 2>::from_floats([[input]], device);

    // Run the model on the input
    model.forward(input)
}

// This runs as a seperate task whenever there is an await
#[embassy_executor::task]
async fn logger_task(driver: Driver<'static, USB>) {
    // This just makes a logger that outputs to serial.
    embassy_usb_logger::run!(1024, log::LevelFilter::Info, driver);
}
