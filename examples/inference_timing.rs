use burn::{
    backend::{
        libtorch::{LibTorch, LibTorchDevice},
        wgpu::{Wgpu, WgpuDevice},
        ndarray::{NdArray, NdArrayDevice},
        Backend,
    },
    tensor::Tensor,
};
use burn_example::{utils::{TimingConfig, measure_inference_time, InferenceMeasurable}};

// Example model that performs matrix multiplication
struct ExampleModel<B: Backend> {
    weights: Tensor<B, 2>,
}

impl<B: Backend> ExampleModel<B> {
    fn new(device: &B::Device) -> Self {
        let weights = Tensor::ones([100, 100]).to_device(device);
        Self { weights }
    }

    fn forward(&self, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        input.matmul(&self.weights)
    }
}

// Implement the InferenceMeasurable trait for our model
impl_inference_measurable!(ExampleModel<B>, Tensor<B, 2>, Tensor<B, 2>);

fn run_benchmark<B: Backend>(device: B::Device) {
    println!("Running benchmark on {}", B::name(&device));
    
    // Create model and input
    let model = ExampleModel::new(&device);
    let input = Tensor::ones([1, 100]).to_device(&device);
    
    // Configure timing parameters
    let config = TimingConfig {
        num_warmup: 10,
        num_iterations: 100,
    };
    
    // Measure inference time
    let results = measure_inference_time(&model, &input, &device, config);
    
    // Print results
    println!("Results:");
    println!("  Mean: {:?}", results.mean);
    println!("  Std Dev: {:?}", results.std_dev);
    println!("  Min: {:?}", results.min);
    println!("  Max: {:?}", results.max);
}

fn main() {
    // Test on CPU with ndarray backend
    println!("\nTesting with NdArray backend (CPU):");
    run_benchmark::<NdArray>(NdArrayDevice::Cpu);
    
    // Test on GPU with WGPU backend if available
    #[cfg(feature = "wgpu")]
    {
        println!("\nTesting with WGPU backend:");
        run_benchmark::<Wgpu>(WgpuDevice::default());
    }
    
    // Test on GPU with LibTorch backend if available
    #[cfg(feature = "libtorch-cuda")]
    {
        println!("\nTesting with LibTorch backend (CUDA):");
        run_benchmark::<LibTorch>(LibTorchDevice::Cuda(0));
    }
    
    // Test on Metal with LibTorch backend if available
    #[cfg(all(feature = "libtorch-metal", target_os = "macos"))]
    {
        println!("\nTesting with LibTorch backend (Metal):");
        run_benchmark::<LibTorch>(LibTorchDevice::Mps);
    }
} 