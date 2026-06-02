use burn::{prelude::*, tensor::Distribution};

fn main() {
    let device1 = Device::cuda(0).autodiff();
    let device2 = Device::cuda(1).autodiff();
    // let device3 = Device::vulkan(Default::default()).autodiff();
    // let device_base = Device::flex().autodiff();

    let linear = nn::LinearConfig::new(8, 8).init(&device2);

    for _ in 0..10 {
        let aa = Tensor::<2>::random([8, 8], Distribution::Default, &device1);
        let rhs = Tensor::<2>::random([8, 8], Distribution::Normal(0.1, 0.6), &device1);
        let lhs = aa + rhs.clone();
        let out = lhs.add(rhs);
        let other = Tensor::<2>::ones([8, 8], &device1);
        let out = out.matmul(other.clone());
        let out = linear.forward(out.to_device(&device2)).to_device(&device1);
        let out = out.sub(other);
        // println!("{out}");
        let loss = out.sum();
        let data = loss.clone().into_data();
        // let loss = loss.to_device(&device_base);
        // println!("{loss}");
        let grad = loss.backward();

        println!("O");
        println!("{data}");
    }
}
