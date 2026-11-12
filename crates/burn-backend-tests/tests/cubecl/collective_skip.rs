use super::*;
use burn_tensor::{Device, TensorData, distributed::ReduceOperation};
use cubek::cubecl::benchmark::CommunicationSkipGuard;

/// One device twice is never a cubecl group, so every collective here is emulated, and with the
/// device skipping communication each member gets the shape it would receive and the all-reduce
/// leaves each member's own values where emulation would sum them.
#[test]
fn an_emulated_collective_skipping_communication_keeps_shapes_and_skips_the_work() {
    let device = Device::default();
    let burn_dispatch::DispatchDevice::Cube(cube_device) = device.as_dispatch() else {
        return;
    };
    let members = || {
        [0.0f32, 10.0]
            .map(|offset| {
                let values: Vec<f32> = (0..6).map(|value| value as f32 + offset).collect();
                TestTensor::<2>::from_data(TensorData::new(values, [3, 2]), &device)
            })
            .to_vec()
    };
    let shapes = |tensors: Vec<TestTensor<2>>| -> Vec<Vec<usize>> {
        tensors
            .into_iter()
            .map(|tensor| tensor.shape().to_vec())
            .collect()
    };
    let collectives = || {
        [
            shapes(TestTensor::all_gather(members(), 0)),
            shapes(TestTensor::reduce_scatter(
                members(),
                0,
                ReduceOperation::Sum,
            )),
            shapes(TestTensor::all_to_all(members(), 0, 1)),
        ]
    };
    let reduced = || -> Vec<TensorData> {
        TestTensor::all_reduce(members(), ReduceOperation::Sum)
            .into_iter()
            .map(TestTensor::into_data)
            .collect()
    };

    let emulated = collectives();
    let summed = reduced();
    let skip = CommunicationSkipGuard::enter(&[cube_device.client()]);
    let skipped = collectives();
    let kept = reduced();
    drop(skip);

    assert_eq!(skipped, emulated);
    assert_eq!(emulated[1], [vec![2, 2], vec![1, 2]], "rows split unevenly");
    let own: Vec<TensorData> = members().into_iter().map(TestTensor::into_data).collect();
    assert_eq!(kept, own);
    assert_ne!(summed, own);
}
