use super::*;
use burn_tensor::{
    TensorPrimitive,
    backend::{
        Backend, Device, DeviceId, DeviceOps,
        distributed::{DistributedBackend, ReduceOperation},
    },
};
use serial_test::serial;

#[test]
#[serial]
fn test_all_reduce() {
    type B = TestBackend;
    // Cuda
    let type_id = 10u16;
    let device_count = <TestBackend as Backend>::device_count(type_id);
    if device_count < 2 {
        return;
    }
    let (device_0, device_1) = create_devices::<<TestBackend as Backend>::Device>(type_id, 2);

    let tensor_0 = TestTensor::<1>::from_data([2.0, 5.0], &device_0).require_grad();
    let tensor_1 = TestTensor::<1>::from_data([4.0, 1.0], &device_1).require_grad();

    let device_ids = vec![device_0, device_1]
        .iter()
        .map(|d| d.id())
        .collect::<Vec<_>>();
    let tensor_2 = B::all_reduce(
        tensor_0.clone().into_primitive().tensor(),
        ReduceOperation::Sum,
        device_ids.clone(),
    );
    let tensor_2: TestTensor<1> =
        TestTensor::new(TensorPrimitive::Float(tensor_2.resolve())).require_grad();
    let grads_0 = tensor_2.backward();

    let tensor_3 = B::all_reduce(
        tensor_1.clone().into_primitive().tensor(),
        ReduceOperation::Sum,
        device_ids,
    );
    let resolved = tensor_3.resolve();
    let DispatchTensor { kind, .. } = resolved.clone();
    println!("kind : {kind:?}");
    let tensor_3: TestTensor<1> = TestTensor::new(TensorPrimitive::Float(resolved)).require_grad();
    let grads_1 = tensor_3.backward();

    println!(
        "tensor_2: {:?}",
        tensor_2.to_data().to_vec::<f32>().unwrap()
    );
    println!(
        "tensor_3: {:?}",
        tensor_3.to_data().to_vec::<f32>().unwrap()
    );

    let grad_0 = tensor_0.grad(&grads_0).unwrap();
    let grad_1 = tensor_1.grad(&grads_1).unwrap();

    println!("tensor_2: {:?}", grad_0.to_data().to_vec::<f32>().unwrap());
    println!("tensor_3: {:?}", grad_1.to_data().to_vec::<f32>().unwrap());
}

fn create_devices<D: Device>(type_id: u16, count: usize) -> (D, D) {
    let devices = (0..count)
        .map(|i| D::from_id(DeviceId::new(type_id, i as u16)))
        .collect::<Vec<_>>();
    (devices[0].clone(), devices[1].clone())
}
