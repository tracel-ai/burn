//! Observed traffic is priced by reading its values back once the run is over, and whatever
//! that readback launches is not the run's work. Run it on a cubecl device (`BURN_DEVICE`).
#![cfg(feature = "collector-tests")]

use burn_instrument::{Bounds, Instrument, ObservedBounds, compute, observing};
use burn_tensor::{Device, Distribution, Int, Tensor};

/// What `#[compute]` expands to, as a crate depending on `burn` sees it.
mod burn {
    pub use burn_instrument as instrument;
}

/// Rows and columns of the input.
const SIDE: usize = 64;

/// A component whose traffic depends on the values it computes.
struct Router;

impl Router {
    /// Declares a floor, so the observed traffic has a component to land on.
    #[compute(bounds = Bounds::default())]
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        if let Some(observing) = observing() {
            // Transposed, so reading it back makes it contiguous first: a kernel the readback
            // launches, after the run.
            let routes: Tensor<2, Int> = x.clone().greater_elem(0.0).int().swap_dims(0, 1);
            observing.observe(ObservedBounds::new(routes, |routes| Bounds {
                read: routes.len() as u64,
                write: 0,
                flops: 0,
            }));
        }
        x.exp()
    }
}

#[test]
fn reading_observed_values_back_launches_nothing_the_pass_records() {
    let device = Device::default();
    let input = Tensor::<2>::random([SIDE, SIDE], Distribution::Default, &device);
    // Clean runs first, so nothing the run launches is a first sighting being tuned.
    let warmed: Vec<Tensor<2>> = (0..5).map(|_| Router.forward(input.clone())).collect();
    device.sync().expect("the device answers before the run");
    core::mem::drop(warmed);

    let breakdown = Instrument::new(&device).breakdown(|| Router.forward(input));

    let names: Vec<&str> = breakdown
        .untagged
        .iter()
        .map(|kernel| kernel.name.as_str())
        .collect();
    assert!(
        names.is_empty(),
        "the readback's kernels were recorded as untagged work of the run: {names:?}"
    );
    assert_eq!(
        breakdown.tree[0].observed,
        Some(Bounds {
            read: (SIDE * SIDE) as u64,
            write: 0,
            flops: 0,
        }),
        "the observed traffic is still priced"
    );
}
