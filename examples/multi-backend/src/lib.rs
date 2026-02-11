use burn::{Engine, EngineDevice, Tensor, backend::Autodiff};

pub fn test_engine() {
    let device = EngineDevice::Cuda(Default::default());
    let device_vulk = EngineDevice::Vulkan(Default::default());

    let zeros_cuda = Tensor::<Engine, 2>::zeros([128, 128], &device);
    println!("{zeros_cuda}");

    let ones = Tensor::<Engine, 2>::ones([128, 128], &device_vulk);
    println!("{ones}");

    let zeros = zeros_cuda.clone().to_device(&device_vulk);
    println!("{zeros}");

    let sum = zeros + ones.clone();
    println!("{sum}");

    // let _invalid = zeros_cuda + ones;

    type EngineAd = Autodiff<BurnBackend>;

    let zeros_cuda = Tensor::<EngineAd, 2>::zeros([128, 128], &device);
    println!("{zeros_cuda}");

    let ones = Tensor::<EngineAd, 2>::ones([128, 128], &device_vulk);
    println!("{ones}");

    let zeros = zeros_cuda.clone().to_device(&device_vulk);
    println!("{zeros}");

    let sum = zeros + ones.clone();
    println!("{sum}");

    let grads = sum.backward();
}
