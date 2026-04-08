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
    run_all_reduce::<TestBackend>();
}

#[test]
#[serial]
fn test_all_reduce_multithread() {
    run_multithread::<TestBackend>();
}

fn run_all_reduce<B: AutodiffBackend + DistributedBackend>() {
    let type_id = 10u16;
    let shape = [4, 4];
    let size = shape[0] * shape[1];

    let device_count = <B as Backend>::device_count(type_id);
    let devices = create_devices::<<B as Backend>::Device>(type_id, device_count);

    let mut rng = rand::rng();

    for i in 0..10 {
        let vec_data: Vec<Vec<f32>> = (0..device_count)
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
        for tensor in tensors {
            let device = tensor.device();
            let output = unsafe {
                B::all_reduce(
                    tensor.into_primitive().tensor(),
                    ReduceOperation::Sum,
                    devices.iter().map(|d| d.id()).collect(),
                )
            };
            let output: Tensor<B, 2, Float> = Tensor::new(TensorPrimitive::Float(output));
            B::sync_collective(&device);
            out_tensors.push(output);
        }

        println!("Expected : {:?}", expected);
        for tensor in out_tensors {
            let data = tensor.flatten::<1>(0, 1).to_data();
            println!("Data : {:?}", data.to_vec::<f32>().unwrap());
            data.assert_approx_eq::<FloatElem>(
                &TensorData::from(expected.as_slice()),
                Tolerance::default(),
            );
        }
    }
}

fn run_multithread<B: AutodiffBackend + DistributedBackend>() {
    let type_id = 10u16;
    let shape = [4, 4];
    let size = shape[0] * shape[1];

    let device_count = <B as Backend>::device_count(type_id);
    let devices = create_devices::<<B as Backend>::Device>(type_id, device_count);
    let device_ids = devices.iter().map(|d| d.id()).collect::<Vec<_>>();

    let handles = devices
        .iter()
        .map(|device| {
            let local_device = device.clone();
            let local_device_ids = device_ids.clone();
            std::thread::spawn(move || {
                for i in 0..10 {
                    let mut rng = rand::rng();
                    let vec_data: Vec<f32> =
                        (0..size).map(|_| rng.random_range(0.0..10.0)).collect();
                    let tensor_data = TensorData::from(vec_data.as_slice());
                    let tensor =
                        Tensor::<B, 1>::from_data(tensor_data, &local_device).reshape(shape);
                    let output = unsafe {
                        B::all_reduce(
                            tensor.into_primitive().tensor(),
                            ReduceOperation::Sum,
                            local_device_ids.clone(),
                        )
                    };
                    let output: Tensor<B, 2, Float> = Tensor::new(TensorPrimitive::Float(output));
                    B::sync_collective(&local_device);

                    let data = output.flatten::<1>(0, 1).to_data();
                    println!(
                        "[{:?}] Data : {:?}",
                        local_device.id(),
                        data.to_vec::<f32>().unwrap()
                    );
                    // data.assert_approx_eq::<FloatElem>(
                    //     &TensorData::from(expected.as_slice()),
                    //     Tolerance::default(),
                    // );
                }
            })
        })
        .collect::<Vec<_>>();

    for h in handles {
        h.join().unwrap();
    }
}

fn create_devices<D: Device>(type_id: u16, count: usize) -> Vec<D> {
    (0..count)
        .map(|i| D::from_id(DeviceId::new(type_id, i as u32)))
        .collect()
}
