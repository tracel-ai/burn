use burn_tensor::{backend::Backend, Bool, Int, Tensor};

use crate::{ConnectedStats, ConnectedStatsOptions, Connectivity, VisionOps};

/// Connected components tensor extensions
pub trait ConnectedComponents<B: Backend> {
    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity - returns a tensor of the component label of each pixel.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components(self, connectivity: Connectivity) -> Tensor<B, 2, Int>;

    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity and collects statistics on each component - returns a tensor of the component
    /// label of each pixel, along with stats collected for each component.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<B, 2, Int>, ConnectedStats<B>);
}

impl<B: Backend + VisionOps<B>> ConnectedComponents<B> for Tensor<B, 2, Bool> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<B, 2, Int> {
        Tensor::from_primitive(B::connected_components(self.into_primitive(), connectivity))
    }

    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<B, 2, Int>, ConnectedStats<B>) {
        let (labels, stats) =
            B::connected_components_with_stats(self.into_primitive(), connectivity, options);
        (Tensor::from_primitive(labels), stats.into())
    }
}
