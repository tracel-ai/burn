use burn_tensor::{backend::Backend, Bool, Int, Tensor};

use crate::{ConnectedStats, ConnectedStatsOptions, Connectivity, VisionOps};

pub trait ConnectedComponents<B: Backend> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<B, 3, Int>;
    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<B, 3, Int>, ConnectedStats<B>);
}

impl<B: Backend + VisionOps<B>> ConnectedComponents<B> for Tensor<B, 4, Bool> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<B, 3, Int> {
        Tensor::from_primitive(B::connected_components(self.into_primitive(), connectivity))
    }

    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<B, 3, Int>, ConnectedStats<B>) {
        let (labels, stats) =
            B::connected_components_with_stats(self.into_primitive(), connectivity, options);
        (Tensor::from_primitive(labels), stats.into())
    }
}
