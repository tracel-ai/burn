//! What an instrumented run across two cards records when each runs part of the model in turn,
//! as a model split by layers does: each card is a member, timed on its own.
#![cfg(any(feature = "sharded-tests-cuda", feature = "sharded-tests-rocm"))]

use burn_instrument::{Bounds, Instrument, MemberSchedule, compute};
use burn_tensor::{Device, DeviceType, Distribution, Tensor};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

const SIDE: usize = 256;

struct Stage;

impl Stage {
    #[compute(bounds = Bounds::default())]
    fn forward(&self, input: Tensor<2>, weight: Tensor<2>) -> Tensor<2> {
        input.matmul(weight).exp()
    }
}

#[test]
fn each_card_of_a_run_split_across_two_is_timed_as_a_member() {
    #[cfg(feature = "sharded-tests-cuda")]
    let runtime = DeviceType::Cuda;
    #[cfg(all(feature = "sharded-tests-rocm", not(feature = "sharded-tests-cuda")))]
    let runtime = DeviceType::Rocm;
    let cards: Vec<Device> = Device::enumerate(runtime)
        .into_vec()
        .into_iter()
        .take(2)
        .collect();
    let [first, second] = cards.as_slice() else {
        panic!("two cards of the runtime are needed");
    };
    let random = |device| Tensor::<2>::random([SIDE, SIDE], Distribution::Default, device);
    let (input, first_weight, second_weight) = (random(first), random(first), random(second));
    let run = || {
        let hidden = Stage.forward(input.clone(), first_weight.clone());
        Stage.forward(hidden.to_device(second), second_weight.clone())
    };
    // Once before the run, so the kernels are compiled.
    drop(run());
    for card in &cards {
        card.sync().expect("the card answers before the run");
    }

    let breakdown = Instrument::for_devices(&cards).breakdown(run);

    assert!(
        !breakdown.tree.is_empty()
            && breakdown
                .tree
                .iter()
                .all(|stage| !stage.total_time().is_zero()),
        "each stage's kernels land under its component: {:?}",
        breakdown.tree
    );
    let per_member = breakdown.per_member.expect("each card is timed apart");
    assert_eq!(per_member.members.len(), 2);
    assert_eq!(per_member.schedule, MemberSchedule::InTurn);
    for (index, member) in per_member.members.iter().enumerate() {
        assert!(!member.on_device.is_zero(), "card {index}: {per_member:?}");
    }
}
