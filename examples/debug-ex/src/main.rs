use burn::{
    prelude::*,
    tensor::{
        DeviceConfig, DeviceType, Distribution, Element,
        distributed::{DistributedConfig, DistributedContext, ReduceOperation},
    },
};

// fn main() {
//     let device1 = Device::cuda(0).autodiff();
//     let device2 = Device::cuda(1).autodiff();
//     // let device3 = Device::vulkan(Default::default()).autodiff();
//     // let device_base = Device::flex().autodiff();

//     let linear = nn::LinearConfig::new(8, 8).init(&device2);

//     for _ in 0..10 {
//         let aa = Tensor::<2>::random([8, 8], Distribution::Default, &device1);
//         let rhs = Tensor::<2>::random([8, 8], Distribution::Normal(0.1, 0.6), &device1);
//         let lhs = aa + rhs.clone();
//         let out = lhs.add(rhs);
//         let other = Tensor::<2>::ones([8, 8], &device1);
//         let out = out.matmul(other.clone());
//         let out = linear.forward(out.to_device(&device2)).to_device(&device1);
//         let out = out.sub(other);
//         // println!("{out}");
//         let loss = out.sum();
//         let data = loss.clone().into_data();
//         // let loss = loss.to_device(&device_base);
//         // println!("{loss}");
//         let grad = loss.backward();

//         println!("O");
//         println!("{data}");
//     }
// }

// fn main() {
//     let mut device = Device::cuda(0);
//     device
//         .configure(
//             DeviceConfig::default()
//                 .float_dtype(<burn::tensor::f16 as Element>::dtype())
//                 .int_dtype(<u32 as Element>::dtype()),
//         )
//         .unwrap();
//     let tensor = Tensor::<1>::from_data([1.0; 5], &device).require_grad();
//     let tensor = tensor.slice([0]);
//     let s = format!("{tensor}");
//     println!("{s}");
// }

fn main() {
    println!("ALL_DEVICES");
    let mut devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    devices.iter_mut().for_each(|d| {
        d.configure(
            DeviceConfig::default()
                .float_dtype(<burn::tensor::f16 as Element>::dtype())
                .int_dtype(<u32 as Element>::dtype()),
        )
        .unwrap();
    });

    let config = DistributedConfig {
        all_reduce_op: ReduceOperation::Sum,
    };
    let _context = DistributedContext::init(devices.clone(), config);

    let input = devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let elem = i as f32;
            Tensor::<1>::from_data([elem, elem, elem, elem, elem], device).require_grad()
        })
        .collect();

    let value: f32 = devices.iter().enumerate().map(|(i, _)| i as f32).sum();
    let grad_value = devices.len() as f32;
    let (output, grads) = compute_gradients(input, ReduceOperation::Sum, devices);
    compare_gradients(output, grads, &[value; 5], &[grad_value; 5]);
}

fn compare_gradients(
    outputs: Vec<Tensor<1>>,
    grads: Vec<Tensor<1>>,
    expected_output: &[f32],
    expected_grads: &[f32],
) {
    println!("In compare gradients");
    for out in outputs {
        let s = format!("{out}");
        println!("out: {s}");
        out.to_data()
            .assert_eq(&TensorData::from(expected_output), false);
    }
    for grad in grads {
        let s = format!("{grad}");
        println!("grad: {s}");
        grad.to_data()
            .assert_eq(&TensorData::from(expected_grads), false);
    }
}

fn compute_gradients(
    tensors: Vec<Tensor<1>>,
    op: ReduceOperation,
    devices: Vec<Device>,
) -> (Vec<Tensor<1>>, Vec<Tensor<1>>) {
    let out = compute_all_reduce(tensors.clone(), op, devices);

    println!("[{:?}] in compute gradients", std::thread::current().id());

    let mut all_grads = vec![];
    for (in_tensor, out_tensor) in tensors.iter().zip(out.clone()) {
        println!("[{:?}] Call backward", std::thread::current().id());
        let grads = out_tensor.backward();
        println!("[{:?}] Call grad", std::thread::current().id());
        all_grads.push(in_tensor.grad(&grads).unwrap());
    }

    (out, all_grads)
}

fn compute_all_reduce(
    tensors: Vec<Tensor<1>>,
    op: ReduceOperation,
    devices: Vec<Device>,
) -> Vec<Tensor<1>> {
    println!("[{:?}] in compute all_reduce", std::thread::current().id());

    let mut out = vec![];
    for tensor in tensors.clone() {
        println!("[{:?}] all_reduce for tensor", std::thread::current().id());

        let out_tensor = burn::tensor::module::all_reduce(tensor, op, devices.clone());

        println!("[{:?}] resolve for tensor", std::thread::current().id());

        let out_tensor = out_tensor.resolve();
        out.push(out_tensor);
    }

    out
}
