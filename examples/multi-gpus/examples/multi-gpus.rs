fn main() {
    #[cfg(feature = "cuda")]
    multi_gpus::run::<burn::backend::Cuda>();
    #[cfg(feature = "rocm")]
    multi_gpus::run::<burn::backend::Rocm>();
    #[cfg(feature = "tch-gpu")]
    multi_gpus::run::<burn::backend::LibTorch>();
}
