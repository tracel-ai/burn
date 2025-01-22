# No Standard Library

In this section, you will learn how to run an onnx inference model on an embedded system, with no standard library support on a Raspberry Pi Pico. This should be universally applicable to other platforms. All the code can be found under the
[examples directory](https://github.com/tracel-ai/burn/tree/main/examples/raspberry-pi-pico).

## Step-by-Step Guide

Let's walk through the process of running an embedded ONNX model:

### Setup
Follow the [embassy guide](https://embassy.dev/book/#_getting_started) for your specific environment. Once setup, you should have something similar to the following.
```
./inference
├── Cargo.lock
├── Cargo.toml
├── build.rs
├── memory.x
└── src
    └── main.rs
```

Some other dependencies have to be added
```toml
[dependencies]
embedded-alloc = "0.5.1" # Only if there is no default allocator for your chip
burn = { version = "0.17", default-features = false, features = ["ndarray"] } # Backend must be ndarray

[build-dependencies]
burn-import = { version = "0.14" } # Used to auto generate the rust code to import the model
```

### Import the Model
Follow the directions to [import models](../import/README.md).

Use the following ModelGen config
```rs
ModelGen::new()
    .input(my_model)
    .out_dir("model/")
    .record_type(RecordType::Bincode)
    .embed_states(true)
    .run_from_script();
```

### Global Allocator
First define a global allocator (if you are on a no_std system without alloc).

```rs
use embedded_alloc::Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
	{
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024; // This is dependent on the model size in memory.
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
    }
}
```

### Define Backend
We are using ndarray, so we just need to define the NdArray backend as usual
```rs
use burn::{backend::NdArray, tensor::Tensor};

type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;
```

Then inside the `main` function add 
```rs
use your_model::Model;

// Get a default device for the backend
let device = BackendDevice::default();

// Create a new model and load the state
let model: Model<Backend> = Model::default();
```

### Running the Model
To run the model, just call it as you would normally
```rs
// Define the tensor
let input = Tensor::<Backend, 2>::from_floats([[input]], &device);

// Run the model on the input
let output = model.forward(input);
```

## Conclusion
Running a model in a no_std environment is pretty much identical to a normal environment. All that is needed is a global allocator. 
