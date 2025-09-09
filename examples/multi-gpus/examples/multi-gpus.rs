fn main() {
    #[cfg(feature = "cuda")]
    multi_gpus::run::<burn::backend::Cuda>(vec![
        burn::backend::cuda::CudaDevice::new(0),
        burn::backend::cuda::CudaDevice::new(1),
        burn::backend::cuda::CudaDevice::new(2),
        burn::backend::cuda::CudaDevice::new(3),
    ]);
}
