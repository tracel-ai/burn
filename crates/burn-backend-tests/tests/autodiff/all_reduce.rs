use super::*;
use burn_dispatch::DispatchDevice;
use burn_tensor::{
    TensorData, TensorPrimitive,
    backend::{
        AutodiffBackend, Backend, Device, DeviceId, DeviceOps,
        distributed::{DistributedBackend, ReduceOperation},
    },
};
use serial_test::serial;

#[test]
#[serial]
fn should_diff_all_reduce_sum() {
    type B = TestBackend;
    let type_id = 10u16;
    let device_count = <B as Backend>::device_count(type_id);
    if device_count < 2 {
        return;
    }

    let (device_0, device_1) = create_devices::<<B as Backend>::Device>(type_id, device_count);

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let (output, grads) = compute_gradients(vec![in_tensor_0, in_tensor_1], ReduceOperation::Sum);
    compare_gradients(output, grads, &[6.0, 6.0], &[2.0, 2.0]);
}

#[test]
#[serial]
fn should_diff_all_reduce_mean() {
    type B = TestBackend;
    let type_id = 10u16;
    let device_count = <B as Backend>::device_count(type_id);
    if device_count < 2 {
        return;
    }

    let (device_0, device_1) = create_devices::<<B as Backend>::Device>(type_id, 2);

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let (output, grads) = compute_gradients(vec![in_tensor_0, in_tensor_1], ReduceOperation::Mean);
    compare_gradients(output, grads, &[3.0, 3.0], &[1.0, 1.0]);
}

#[test]
#[serial]
fn should_diff_all_reduce_complex_1() {
    type B = TestBackend;
    let type_id = 10u16;
    let device_count = <B as Backend>::device_count(type_id);
    if device_count < 2 {
        return;
    }

    let (device_0, device_1) = create_devices::<<B as Backend>::Device>(type_id, 2);

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let [out_tensor_0, out_tensor_1] = &compute_all_reduce(
        vec![in_tensor_0.clone(), in_tensor_1.clone()],
        ReduceOperation::Sum,
    )[..] else {
        panic!("should have 2 tensors")
    };

    let out_tensor_1 = out_tensor_1.clone().mul_scalar(4.0);

    let grads_0 = out_tensor_0.backward();
    let grads_1 = out_tensor_1.backward();

    let grad_0 = in_tensor_0.grad(&grads_0).unwrap();
    let grad_1 = in_tensor_1.grad(&grads_1).unwrap();

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
    type B = TestBackend;
    let type_id = 10u16;
    let device_count = <B as Backend>::device_count(type_id);
    let devices = (0..device_count)
        .map(|i| {
            burn_dispatch::DispatchDevice::autodiff(<B as Backend>::Device::from_id(DeviceId::new(
                type_id, i as u16,
            )))
        })
        .collect::<Vec<_>>();

    let input = devices
        .iter()
        .map(|device| {
            let elem = device.id().index_id as f32;
            TestTensor::<1>::from_data([elem, elem], device).require_grad()
        })
        .collect();

    let value: f32 = devices
        .iter()
        .map(|device| device.id().index_id as f32)
        .sum();
    let grad_value = devices.len() as f32;
    let (output, grads) = compute_gradients(input, ReduceOperation::Sum);
    compare_gradients(output, grads, &[value, value], &[grad_value, grad_value]);
}

fn compare_gradients<B: AutodiffBackend>(
    outputs: Vec<Tensor<B, 1>>,
    grads: Vec<Tensor<B::InnerBackend, 1>>,
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

fn compute_gradients<B: AutodiffBackend + DistributedBackend>(
    tensors: Vec<Tensor<B, 1>>,
    op: ReduceOperation,
) -> (Vec<Tensor<B, 1>>, Vec<Tensor<B::InnerBackend, 1>>) {
    let out = compute_all_reduce(tensors.clone(), op);

    let mut all_grads = vec![];
    for (in_tensor, out_tensor) in tensors.iter().zip(out.clone()) {
        let grads = out_tensor.backward();
        all_grads.push(in_tensor.grad(&grads).unwrap());
    }

    (out, all_grads)
}

fn compute_all_reduce<B: AutodiffBackend + DistributedBackend>(
    tensors: Vec<Tensor<B, 1>>,
    op: ReduceOperation,
) -> Vec<Tensor<B, 1>> {
    let device_ids = tensors
        .iter()
        .map(|tensor| tensor.device().id())
        .collect::<Vec<_>>();

    let mut out = vec![];
    for tensor in tensors.clone() {
        let out_tensor = B::all_reduce(tensor.into_primitive().tensor(), op, device_ids.clone());
        let resolved = out_tensor.resolve();
        let out_tensor = Tensor::<B, 1>::new(TensorPrimitive::Float(resolved));
        out.push(out_tensor);
    }

    out
}

fn create_devices<D: Device>(type_id: u16, count: usize) -> (DispatchDevice, DispatchDevice)
where
    DispatchDevice: From<D>,
{
    let devices = (0..count)
        .map(|i| D::from_id(DeviceId::new(type_id, i as u16)))
        .collect::<Vec<_>>();
    (
        burn_dispatch::DispatchDevice::autodiff(devices[0].clone()),
        burn_dispatch::DispatchDevice::autodiff(devices[1].clone()),
    )
}
