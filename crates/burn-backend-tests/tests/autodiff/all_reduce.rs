use super::*;
use burn_dispatch::{DispatchDevice, DispatchTensor};
use burn_tensor::{
    TensorData, TensorPrimitive,
    backend::{
        Backend, Device, DeviceId, DeviceOps,
        distributed::{DistributedBackend, ReduceOperation},
    },
};
use serial_test::serial;

#[test]
#[serial]
fn should_diff_all_reduce_sum() {
    type B = TestBackend;
    let type_id = 10u16;
    let device_count = <TestBackend as Backend>::device_count(type_id);
    if device_count < 2 {
        return;
    }

    let (device_0, device_1) = create_devices::<<TestBackend as Backend>::Device>(type_id, 2);

    let in_tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let in_tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let device_ids = vec![device_0, device_1]
        .iter()
        .map(|d| d.id())
        .collect::<Vec<_>>();
    let out_tensor_0 = B::all_reduce(
        in_tensor_0.clone().into_primitive().tensor(),
        ReduceOperation::Sum,
        device_ids.clone(),
    );
    let resolved = out_tensor_0.resolve();
    let out_tensor_0: TestTensor<1> = TestTensor::new(TensorPrimitive::Float(resolved));

    let out_tensor_1 = B::all_reduce(
        in_tensor_1.clone().into_primitive().tensor(),
        ReduceOperation::Sum,
        device_ids,
    );
    let resolved = out_tensor_1.resolve();
    let out_tensor_1: TestTensor<1> = TestTensor::new(TensorPrimitive::Float(resolved));

    // let tensor_2 = out_tensor_0.clone().mul_scalar(4.0);

    let grads_0 = out_tensor_0.backward();
    let grads_1 = out_tensor_1.backward();

    let grad_0 = in_tensor_0.grad(&grads_0).unwrap();
    let grad_1 = in_tensor_1.grad(&grads_1).unwrap();

    grad_0
        .to_data()
        .assert_eq(&TensorData::from([2.0, 2.0]), false);
    grad_1
        .to_data()
        .assert_eq(&TensorData::from([2.0, 2.0]), false);
    out_tensor_0
        .to_data()
        .assert_eq(&TensorData::from([6.0, 6.0]), false);
    out_tensor_1
        .to_data()
        .assert_eq(&TensorData::from([6.0, 6.0]), false);
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
