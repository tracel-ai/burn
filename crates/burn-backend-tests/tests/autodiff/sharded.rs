use std::sync::mpsc::SyncSender;

use burn_autodiff::Autodiff;
use burn_collective::{CollectiveConfig, register, reset_collective};
use burn_tensor::{
    Tensor, TensorData, Tolerance,
    backend::{AllReduceStrategy, AutodiffBackend, Backend, PeerId, ReduceOperation},
};
use rand::rngs::StdRng;
use rand::{SeedableRng, rngs::SysRng};
use serial_test::serial;

pub type TestBackend = burn_ndarray::NdArray<f32>;
pub type TestAutodiffBackend = Autodiff<TestBackend>;

// TODO: Tests still relevant?

pub fn run_peer_sharded<B, const D_OUT: usize>(
    id: PeerId,
    config: CollectiveConfig,
    input: TensorData,
    op: ReduceOperation,
    output: SyncSender<Tensor<<B as AutodiffBackend>::InnerBackend, 1>>,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, D_OUT>,
) where
    B: AutodiffBackend,
{
    let device = B::Device::default();

    register::<<B as AutodiffBackend>::InnerBackend>(id, device.clone(), config).unwrap();

    let input_tensor = Tensor::<B, 1>::from_data(input, &device)
        .require_grad()
        .set_sharded_params(id, op, None);
    let out_tensor = transformation(input_tensor.clone());
    let grads = out_tensor.backward();

    let tensor_grad = input_tensor.grad(&grads).unwrap();

    output.send(tensor_grad).unwrap();
}

fn generate_random_input_autodiff<B, const D_OUT: usize>(
    shape: Vec<usize>,
    op: ReduceOperation,
    thread_count: usize,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, D_OUT>,
) -> (Vec<TensorData>, TensorData)
where
    B: AutodiffBackend,
{
    let input: Vec<TensorData> = (0..thread_count)
        .map(|_| {
            TensorData::random::<f32, _, _>(
                shape.clone(),
                burn_tensor::Distribution::Default,
                &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
            )
        })
        .collect();

    let device = <B as Backend>::Device::default();

    let mut expected_tensor =
        Tensor::<<B as AutodiffBackend>::InnerBackend, 1>::zeros(shape, &device);
    for item in input.iter().take(thread_count as usize) {
        let input_tensor = Tensor::<B, 1>::from_data(item.clone(), &device).require_grad();
        let out_tensor = transformation(input_tensor.clone());
        let grads = out_tensor.backward();
        let grads = input_tensor.grad(&grads).unwrap();
        expected_tensor = expected_tensor.add(grads);
    }
    if op == ReduceOperation::Mean {
        expected_tensor = expected_tensor.div_scalar(thread_count as u32);
    }

    let expected = expected_tensor.to_data();

    (input, expected)
}

fn test_sharded_backward<B, const D_OUT: usize>(
    op: ReduceOperation,
    tensor_size: usize,
    transformation: fn(Tensor<B, 1>) -> Tensor<B, D_OUT>,
) where
    B: AutodiffBackend,
{
    reset_collective::<<TestAutodiffBackend as AutodiffBackend>::InnerBackend>();
    let device_count = 4;

    let (send, recv) = std::sync::mpsc::sync_channel(32);

    let shape = vec![tensor_size];

    let (input, expected) = generate_random_input_autodiff(shape, op, device_count, transformation);

    let config = CollectiveConfig::default()
        .with_num_devices(device_count)
        .with_local_all_reduce_strategy(AllReduceStrategy::Centralized);

    for id in 0..device_count {
        let send = send.clone();
        let input = input[id as usize].clone();

        std::thread::spawn({
            let config = config.clone();
            move || run_peer_sharded::<B, _>(id.into(), config, input, op, send, transformation)
        });
    }

    let first = recv.recv().unwrap().to_data();
    for _ in 1..device_count {
        let tensor = recv.recv().unwrap();
        tensor.to_data().assert_eq(&first, true);
    }

    let tol: Tolerance<f32> = Tolerance::balanced();
    expected.assert_approx_eq(&first, tol);
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_identity() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| tensor);
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_identity() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| tensor);
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_mul_scalar() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| {
        tensor.mul_scalar(3)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_mul_scalar() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| {
        tensor.mul_scalar(3)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_multi_node_1() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| {
        let tensor1 = tensor.clone().mul_scalar(3);
        let tensor2 = tensor.mul_scalar(2);
        Tensor::cat(vec![tensor1, tensor2], 0)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_multi_node_1() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| {
        let tensor1 = tensor.clone().mul_scalar(3);
        let tensor2 = tensor.mul_scalar(2);
        Tensor::cat(vec![tensor1, tensor2], 0)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_residual() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| {
        let path1 = tensor.clone().mul_scalar(2.0);
        let path2 = tensor;
        path1.add(path2)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_residual() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| {
        let path1 = tensor.clone().mul_scalar(2.0);
        let path2 = tensor;
        path1.add(path2)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_reshape() {
    test_sharded_backward::<TestAutodiffBackend, 2>(ReduceOperation::Sum, 4, |tensor| {
        tensor.reshape([2, 2])
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_reshape() {
    test_sharded_backward::<TestAutodiffBackend, 2>(ReduceOperation::Mean, 4, |tensor| {
        tensor.reshape([2, 2])
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_activation() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| {
        burn_tensor::activation::relu(tensor)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_activation() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| {
        burn_tensor::activation::relu(tensor)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_sum_diamond_graph() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Sum, 4, |tensor| {
        let root = tensor.mul_scalar(0.5);
        let left = root.clone().exp();
        let right = root.mul_scalar(4.0);
        Tensor::cat(vec![left, right], 0)
    });
}

#[test]
#[serial]
pub fn test_sharded_backward_mean_diamond_graph() {
    test_sharded_backward::<TestAutodiffBackend, _>(ReduceOperation::Mean, 4, |tensor| {
        let root = tensor.mul_scalar(0.5);
        let left = root.clone().exp();
        let right = root.mul_scalar(4.0);
        Tensor::cat(vec![left, right], 0)
    });
}
