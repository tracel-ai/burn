use super::*;
use burn_tensor::{
    Device, DeviceType, TensorData,
    distributed::{DistributedConfig, DistributedContext, ReduceOperation},
};
use serial_test::serial;

// TODO

#[test]
#[serial]
fn should_diff_all_reduce_sum() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    if devices.len() < 2 {
        return;
    }
    let (device_0, device_1) = (devices[0].clone(), devices[1].clone());

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let (output, grads) = compute_gradients(
        vec![in_tensor_0, in_tensor_1],
        ReduceOperation::Sum,
        vec![device_0, device_1],
    );
    compare_gradients(output, grads, &[6.0, 6.0], &[2.0, 2.0]);
}

#[test]
#[serial]
fn should_diff_all_reduce_mean() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    if devices.len() < 2 {
        return;
    }
    let (device_0, device_1) = (devices[0].clone(), devices[1].clone());

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let (output, grads) = compute_gradients(
        vec![in_tensor_0, in_tensor_1],
        ReduceOperation::Mean,
        vec![device_0, device_1],
    );
    compare_gradients(output, grads, &[3.0, 3.0], &[1.0, 1.0]);
}

#[test]
#[serial]
fn should_diff_all_reduce_complex_1() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    if devices.len() < 2 {
        return;
    }
    let (device_0, device_1) = (devices[0].clone(), devices[1].clone());

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let config = DistributedConfig {
        all_reduce_op: ReduceOperation::Sum,
    };
    let _context = DistributedContext::init(vec![device_0.clone(), device_1.clone()], config);
    let [out_tensor_0, out_tensor_1] = &compute_all_reduce(
        vec![in_tensor_0.clone(), in_tensor_1.clone()],
        ReduceOperation::Sum,
        vec![device_0, device_1],
    )[..] else {
        panic!("should have 2 tensors")
    };

    let out_tensor_1 = out_tensor_1.clone().mul_scalar(4.0);

    let grads_0 = out_tensor_0.backward();
    let grads_1 = out_tensor_1.backward();

    let grad_0 = in_tensor_0.grad(&grads_0).unwrap();
    let grad_1 = in_tensor_1.grad(&grads_1).unwrap();

    out_tensor_0.device().sync().unwrap();

    out_tensor_0
        .to_data()
        .assert_eq(&TensorData::from([6.0, 6.0]), false);
    out_tensor_1
        .to_data()
        .assert_eq(&TensorData::from([24.0, 24.0]), false);
    grad_0
        .to_data()
        .assert_eq(&TensorData::from([5.0, 5.0]), false);
    grad_1
        .to_data()
        .assert_eq(&TensorData::from([5.0, 5.0]), false);
}

#[test]
#[serial]
fn should_diff_all_reduce_all_devices() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();

    let input = devices
        .iter()
        .enumerate()
        .map(|(i, device)| {
            let elem = i as f32;
            TestTensor::<1>::from_data([elem, elem], device).require_grad()
        })
        .collect();

    let value: f32 = devices.iter().enumerate().map(|(i, _)| i as f32).sum();
    let grad_value = devices.len() as f32;
    let (output, grads) = compute_gradients(input, ReduceOperation::Sum, devices);
    compare_gradients(output, grads, &[value, value], &[grad_value, grad_value]);
}

fn compare_gradients(
    outputs: Vec<Tensor<1>>,
    grads: Vec<Tensor<1>>,
    expected_output: &[f32],
    expected_grads: &[f32],
) {
    for out in outputs {
        out.to_data()
            .assert_eq(&TensorData::from(expected_output), false);
    }
    for grad in grads {
        grad.to_data()
            .assert_eq(&TensorData::from(expected_grads), false);
    }
}

fn compute_gradients(
    tensors: Vec<Tensor<1>>,
    op: ReduceOperation,
    devices: Vec<Device>,
) -> (Vec<Tensor<1>>, Vec<Tensor<1>>) {
    let config = DistributedConfig { all_reduce_op: op };
    let _context = DistributedContext::init(devices.clone(), config);

    let out = compute_all_reduce(tensors.clone(), op, devices);

    let mut all_grads = vec![];
    for (in_tensor, out_tensor) in tensors.iter().zip(out.clone()) {
        let grads = out_tensor.backward();
        all_grads.push(in_tensor.grad(&grads).unwrap());
    }

    (out, all_grads)
}

fn compute_all_reduce(
    tensors: Vec<Tensor<1>>,
    op: ReduceOperation,
    devices: Vec<Device>,
) -> Vec<Tensor<1>> {
    let mut out = vec![];
    for tensor in tensors.clone() {
        let out_tensor = burn_tensor::module::all_reduce(tensor, op, devices.clone());
        let out_tensor = out_tensor.resolve();
        out.push(out_tensor);
    }

    out
}
