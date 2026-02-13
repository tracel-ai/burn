use burn::{Device, Dispatch, Tensor, backend::Autodiff};

pub fn test_dispatch() {
    let device = Device::Cuda(Default::default());
    let device_cpu = Device::Cpu(Default::default());

    let zeros_cuda = Tensor::<Dispatch, 2>::zeros([128, 128], &device);
    println!("{zeros_cuda}");

    let ones = Tensor::<Dispatch, 2>::ones([128, 128], &device_cpu);
    println!("{ones}");

    let zeros = zeros_cuda.clone().to_device(&device_cpu);
    println!("{zeros}");

    let sum = zeros + ones.clone();
    println!("{sum}");

    // let _invalid = zeros_cuda + ones;

    type EngineAd = Autodiff<Dispatch>;

    let zeros_cuda = Tensor::<EngineAd, 2>::zeros([128, 128], &device);
    println!("{zeros_cuda}");

    let ones = Tensor::<EngineAd, 2>::ones([128, 128], &device_cpu);
    println!("{ones}");

    let zeros = zeros_cuda.clone().to_device(&device_cpu);
    println!("{zeros}");

    let sum = zeros + ones.clone();
    println!("{sum}");

    let _grads = sum.backward();
}
