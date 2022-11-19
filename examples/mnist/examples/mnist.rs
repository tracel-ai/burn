#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
))]
mod ndarray {
    use burn::tensor::backend::ADBackendDecorator;
    use burn_ndarray::{NdArrayBackend, NdArrayDevice};
    use mnist::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<ADBackendDecorator<NdArrayBackend<f32>>>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::tensor::backend::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};
    use mnist::training;

    pub fn run() {
        let device = TchDevice::Cuda(0);
        training::run::<ADBackendDecorator<TchBackend<burn::tensor::f16>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::tensor::backend::ADBackendDecorator;
    use burn_tch::{TchBackend, TchDevice};
    use mnist::training;

    pub fn run() {
        let device = TchDevice::Cpu;
        training::run::<ADBackendDecorator<TchBackend<f32>>>(device);
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
}
