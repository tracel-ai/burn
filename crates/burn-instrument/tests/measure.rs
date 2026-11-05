//! A measurement taken on a cubecl device (`BURN_DEVICE`) carries what a report scores it
//! against beside the run it breaks down.
#![cfg(feature = "collector-tests")]

use burn_instrument::{Bounds, Instrument, compute};
use burn_tensor::{Device, Distribution, Tensor};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

/// Rows and columns of the operands.
const SIDE: usize = 64;

struct Square;

impl Square {
    #[compute(bounds = Bounds::default())]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.clone().matmul(x)
    }
}

#[test]
fn a_measurement_carries_its_peaks_and_clean_wall_beside_the_run() {
    let device = Device::default();
    let input = Tensor::<2>::random([SIDE, SIDE], Distribution::Default, &device);
    device
        .sync()
        .expect("the device answers before the measurement");

    let measurement = Instrument::new(&device)
        .warmup(2)
        .samples(3)
        .measure(|| Square.forward(input.clone()));

    let peaks = measurement.peaks.expect("probed before the clean runs");
    assert!(
        peaks.copy_bytes_per_s.is_some(),
        "a cubecl device has a copy probe"
    );
    assert!(measurement.clean_wall.is_some_and(|wall| !wall.is_zero()));
    let [square] = measurement.breakdown.tree.as_slice() else {
        panic!("one component: {:?}", measurement.breakdown.tree);
    };
    assert_eq!(square.label, "square");
    assert_eq!(square.entries, 1, "only the instrumented run is recorded");
    assert!(
        square.kernels.iter().any(|kernel| !kernel.total.is_zero()),
        "the matmul ran and was timed: {:?}",
        square.kernels
    );
}
