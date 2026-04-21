use burn_tensor::backend::DeviceOps;
use burn_tensor::backend::{Device, DeviceId};
use burn_tensor::{Float, TensorPrimitive, Tolerance};
use burn_tensor::{
    TensorData,
    backend::{
        AutodiffBackend, Backend,
        distributed::{DistributedBackend, ReduceOperation},
    },
};
use rand::RngExt;
use serial_test::serial;

use super::*;

#[test]
#[serial]
fn test_all_reduce() {
    // Cuda
    let type_id = 10u16;
    let device_count = <TestBackend as Backend>::device_count(type_id);
    let devices = create_devices::<<TestBackend as Backend>::Device>(type_id, device_count);

    let shape = [20, 20];
    run_all_reduce::<TestBackend>(devices, 100, shape);
}

#[test]
#[serial]
fn test_all_reduce_multithread() {
    // Cuda
    let type_id = 10u16;
    let device_count = <TestBackend as Backend>::device_count(type_id);
    let devices = create_devices::<<TestBackend as Backend>::Device>(type_id, device_count);

    let shape = [20, 20];
    run_multithread::<TestBackend>(devices, 100, shape);
}

fn run_all_reduce<B: AutodiffBackend + DistributedBackend>(
    devices: Vec<B::Device>,
    num_iterations: usize,
    shape: [usize; 2],
) {
    let mut rng = rand::rng();
    let size = shape[0] * shape[1];

    for _ in 0..num_iterations {
        let vec_data: Vec<Vec<f32>> = (0..devices.len())
            .map(|_| (0..size).map(|_| rng.random_range(0.0..10.0)).collect())
            .collect();
        let expected: Vec<f32> = (0..size)
            .map(|i| vec_data.iter().map(|v| v[i]).sum::<f32>())
            .collect();
        let tensors: Vec<Tensor<B, 2>> = vec_data
            .iter()
            .zip(devices.clone())
            .map(|(data, device)| {
                let tensor_data = TensorData::from(data.as_slice());
                let tensor = Tensor::<B, 1>::from_data(tensor_data, &device);
                tensor.reshape(shape)
            })
            .collect();

        let mut out_tensors = vec![];
        for tensor in tensors.clone() {
            let output = B::all_reduce(
                tensor.into_primitive().tensor(),
                ReduceOperation::Sum,
                devices.iter().map(|d| d.id()).collect(),
            );
            let output: Tensor<B, 2, Float> = Tensor::new(TensorPrimitive::Float(output.resolve()));
            out_tensors.push(output);
        }

        for tensor in out_tensors {
            let data = tensor.flatten::<1>(0, 1).to_data();
            data.assert_approx_eq::<FloatElem>(
                &TensorData::from(expected.as_slice()),
                Tolerance::default(),
            );
        }
    }
}

fn run_multithread<B: AutodiffBackend + DistributedBackend>(
    devices: Vec<B::Device>,
    num_iterations: usize,
    shape: [usize; 2],
) {
    let size = shape[0] * shape[1];
    let device_ids = devices.iter().map(|d| d.id()).collect::<Vec<_>>();
    let num_devices = devices.len();

    let (expected_sender, expected_receiver) = std::sync::mpsc::channel();
    let (actual_sender, actual_receiver) = std::sync::mpsc::channel();
    let handles = devices
        .iter()
        .map(|device| {
            let local_device = device.clone();
            let local_device_ids = device_ids.clone();
            let local_expected_sender = expected_sender.clone();
            let local_actual_sender = actual_sender.clone();
            std::thread::spawn(move || {
                for _ in 0..num_iterations {
                    let mut rng = rand::rng();
                    let vec_data: Vec<f32> =
                        (0..size).map(|_| rng.random_range(0.0..10.0)).collect();
                    local_expected_sender.send(vec_data.clone()).unwrap();

                    let tensor_data = TensorData::from(vec_data.as_slice());
                    let tensor = Tensor::<B, 1>::from_data(tensor_data, &local_device)
                        .reshape(shape.clone());
                    let output = B::all_reduce(
                        tensor.into_primitive().tensor(),
                        ReduceOperation::Sum,
                        local_device_ids.clone(),
                    );
                    let output: Tensor<B, 2, Float> =
                        Tensor::new(TensorPrimitive::Float(output.resolve()));

                    let data = output.flatten::<1>(0, 1).to_data();
                    local_actual_sender.send(data).unwrap();
                }
            })
        })
        .collect::<Vec<_>>();

    for _ in 0..num_iterations {
        let mut expected_list = vec![];
        for _ in 0..num_devices {
            expected_list.push(expected_receiver.recv().unwrap());
        }
        let expected: Vec<f32> = (0..size)
            .map(|i| expected_list.iter().map(|v| v[i]).sum::<f32>())
            .collect();

        for _ in 0..num_devices {
            let data = actual_receiver.recv().unwrap();
            data.assert_approx_eq::<FloatElem>(
                &TensorData::from(expected.as_slice()),
                Tolerance::default(),
            );
        }
    }

    for h in handles {
        h.join().unwrap();
    }
}

fn create_devices<D: Device>(type_id: u16, count: usize) -> Vec<D> {
    (0..count)
        .map(|i| D::from_id(DeviceId::new(type_id, i as u16)))
        .collect()
}
