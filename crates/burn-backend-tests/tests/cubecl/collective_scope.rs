use super::*;
use burn_tensor::{Device, TensorData, distributed::ReduceOperation};
use cubek::cubecl::{
    benchmark::CommunicationSkipGuard,
    logging::{
        CollectivePath, Communication, Execution, ExecutionKind, ExecutionObservation,
        ExecutionObserver,
    },
    server::Collective,
};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Recorded(Mutex<Vec<Execution>>);

impl ExecutionObserver for Recorded {
    fn issued(&self, execution: &Execution) {
        self.0.lock().unwrap().push(execution.clone());
    }
}

/// An emulated collective is one communication over every member, the parent of the kernels it
/// runs, so a profile charges them to it rather than to nothing; skipped, it is still issued.
#[test]
#[serial_test::serial(communication_skip)]
fn an_emulated_collective_parents_the_kernels_it_runs() {
    let device = Device::default();
    let burn_dispatch::DispatchDevice::Cube(cube_device) = device.as_dispatch() else {
        return;
    };
    // Five rows of three f32 are a size no other collective here uses: the observer sees the
    // whole process.
    const BYTES: u64 = 5 * 3 * 4;
    let members = || {
        [0.0f32, 10.0]
            .map(|offset| {
                let values: Vec<f32> = (0..15).map(|value| value as f32 + offset).collect();
                TestTensor::<2>::from_data(TensorData::new(values, [5, 3]), &device)
            })
            .to_vec()
    };
    let observed = |members: Vec<TestTensor<2>>| -> Vec<Execution> {
        let recorded = Arc::new(Recorded::default());
        let watching = ExecutionObservation::new(recorded.clone());
        let reduced = TestTensor::all_reduce(members, ReduceOperation::Sum);
        drop(watching);
        drop(reduced);
        recorded.0.lock().unwrap().clone()
    };
    let collective = |issued: &[Execution]| -> (Execution, Vec<Execution>) {
        let [collective] = issued
            .iter()
            .filter(|execution| {
                matches!(
                    &execution.kind,
                    ExecutionKind::Communication(Communication::Collective {
                        operation: Collective::AllReduce,
                        bytes: BYTES,
                        ..
                    })
                )
            })
            .collect::<Vec<_>>()[..]
        else {
            panic!("not one all-reduce in {issued:?}");
        };
        let children = issued
            .iter()
            .filter(|execution| execution.parent == Some(collective.id))
            .cloned()
            .collect();
        (collective.clone(), children)
    };
    let service = cube_device.client().service_id();

    let (emulated, children) = collective(&observed(members()));
    let ExecutionKind::Communication(Communication::Collective {
        members: ids, path, ..
    }) = emulated.kind
    else {
        unreachable!()
    };
    assert_eq!(ids, [service, service]);
    assert!(matches!(path, CollectivePath::Emulated { .. }), "{path:?}");
    assert!(
        children
            .iter()
            .any(|child| matches!(child.kind, ExecutionKind::Kernel { .. })),
        "{children:?}"
    );

    let tensors = members();
    let skip = CommunicationSkipGuard::enter(&[cube_device.client()]);
    let (skipped, children) = collective(&observed(tensors));
    drop(skip);
    let ExecutionKind::Communication(Communication::Collective { path, .. }) = skipped.kind else {
        unreachable!()
    };
    assert_eq!(path, CollectivePath::Skipped);
    assert!(children.is_empty(), "{children:?}");
}
