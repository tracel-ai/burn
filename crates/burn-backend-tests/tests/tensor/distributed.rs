use burn_tensor::Tolerance;
use burn_tensor::distributed::ReduceOperation;
use burn_tensor::module::all_reduce;
use burn_tensor::{Device, DeviceType, TensorData};
use rand::RngExt;
use serial_test::serial;

use super::*;

#[test]
#[serial]
fn test_all_reduce() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    let shape = [20, 20];
    run_all_reduce(devices, 100, shape);
}

#[test]
#[serial]
fn test_all_reduce_multithread() {
    let devices = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    let shape = [20, 20];
    run_multithread(devices, 100, shape);
}

fn run_all_reduce(devices: Vec<Device>, num_iterations: usize, shape: [usize; 2]) {
    let mut rng = rand::rng();
    let size = shape[0] * shape[1];

    for _ in 0..num_iterations {
        let vec_data: Vec<Vec<f32>> = (0..devices.len())
            .map(|_| (0..size).map(|_| rng.random_range(0.0..10.0)).collect())
            .collect();
        let expected: Vec<f32> = (0..size)
            .map(|i| vec_data.iter().map(|v| v[i]).sum::<f32>())
            .collect();
        let tensors: Vec<Tensor<2>> = vec_data
            .iter()
            .zip(devices.clone())
            .map(|(data, device)| {
                let tensor_data = TensorData::from(data.as_slice());
                let tensor = Tensor::<1>::from_data(tensor_data, &device);
                tensor.reshape(shape)
            })
            .collect();

        let mut out_tensors = vec![];
        for tensor in tensors.clone() {
            let output = all_reduce(tensor, ReduceOperation::Sum, devices.clone());
            let output = output.resolve();
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

fn run_multithread(devices: Vec<Device>, num_iterations: usize, shape: [usize; 2]) {
    let size = shape[0] * shape[1];
    let num_devices = devices.len();

    let (expected_sender, expected_receiver) = std::sync::mpsc::channel();
    let (actual_sender, actual_receiver) = std::sync::mpsc::channel();
    let handles = devices
        .iter()
        .map(|device| {
            let local_device = device.clone();
            let local_devices = devices.clone();
            let local_expected_sender = expected_sender.clone();
            let local_actual_sender = actual_sender.clone();
            std::thread::spawn(move || {
                for _ in 0..num_iterations {
                    let mut rng = rand::rng();
                    let vec_data: Vec<f32> =
                        (0..size).map(|_| rng.random_range(0.0..10.0)).collect();
                    local_expected_sender.send(vec_data.clone()).unwrap();

                    let tensor_data = TensorData::from(vec_data.as_slice());
                    let tensor =
                        Tensor::<1>::from_data(tensor_data, &local_device).reshape(shape.clone());
                    let output = all_reduce(tensor, ReduceOperation::Sum, local_devices.clone());
                    let output = output.resolve();

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
