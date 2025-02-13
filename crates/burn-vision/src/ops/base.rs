use crate::backends::cpu;
use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
    Int, Tensor,
};

/// Connected components connectivity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Connectivity {
    /// Four-connected (only connected in cardinal directions)
    Four,
    /// Eight-connected (connected if any of the surrounding 8 pixels are in the foreground)
    Eight,
}

/// Which stats should be enabled for `connected_components_with_stats`.
/// Currently only used by the GPU implementation to save on atomic operations for unneeded stats.
///
/// Disabled stats are aliased to the labels tensor
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnectedStatsOptions {
    /// Whether to enable bounding boxes
    pub bounds_enabled: bool,
    /// Whether to enable the max label
    pub max_label_enabled: bool,
    /// Whether labels must be contiguous starting at 1
    pub compact_labels: bool,
}

/// Stats collected by the connected components analysis
///
/// Disabled analyses may be aliased to labels
#[derive(Clone, Debug)]
pub struct ConnectedStats<B: Backend> {
    /// Total area of each component
    pub area: Tensor<B, 1, Int>,
    /// Topmost y coordinate in the component
    pub top: Tensor<B, 1, Int>,
    /// Leftmost x coordinate in the component
    pub left: Tensor<B, 1, Int>,
    /// Rightmost x coordinate in the component
    pub right: Tensor<B, 1, Int>,
    /// Bottommost y coordinate in the component
    pub bottom: Tensor<B, 1, Int>,
    /// Scalar tensor of the max label
    pub max_label: Tensor<B, 1, Int>,
}

/// Primitive version of [`ConnectedStats`], to be returned by the backend
pub struct ConnectedStatsPrimitive<B: Backend> {
    /// Total area of each component
    pub area: IntTensor<B>,
    /// Leftmost x coordinate in the component
    pub left: IntTensor<B>,
    /// Topmost y coordinate in the component
    pub top: IntTensor<B>,
    /// Rightmost x coordinate in the component
    pub right: IntTensor<B>,
    /// Bottommost y coordinate in the component
    pub bottom: IntTensor<B>,
    /// Scalar tensor of the max label
    pub max_label: IntTensor<B>,
}

impl<B: Backend> From<ConnectedStatsPrimitive<B>> for ConnectedStats<B> {
    fn from(value: ConnectedStatsPrimitive<B>) -> Self {
        ConnectedStats {
            area: Tensor::from_primitive(value.area),
            top: Tensor::from_primitive(value.top),
            left: Tensor::from_primitive(value.left),
            right: Tensor::from_primitive(value.right),
            bottom: Tensor::from_primitive(value.bottom),
            max_label: Tensor::from_primitive(value.max_label),
        }
    }
}

impl<B: Backend> ConnectedStats<B> {
    /// Convert a connected stats into the corresponding primitive
    pub fn into_primitive(self) -> ConnectedStatsPrimitive<B> {
        ConnectedStatsPrimitive {
            area: self.area.into_primitive(),
            top: self.top.into_primitive(),
            left: self.left.into_primitive(),
            right: self.right.into_primitive(),
            bottom: self.bottom.into_primitive(),
            max_label: self.max_label.into_primitive(),
        }
    }
}

impl Default for ConnectedStatsOptions {
    fn default() -> Self {
        Self::all()
    }
}

impl ConnectedStatsOptions {
    /// Don't collect any stats
    pub fn none() -> Self {
        Self {
            bounds_enabled: false,
            max_label_enabled: false,
            compact_labels: false,
        }
    }

    /// Collect all stats
    pub fn all() -> Self {
        Self {
            bounds_enabled: true,
            max_label_enabled: true,
            compact_labels: true,
        }
    }
}

/// Vision operations, implemented by each backend
pub trait VisionOps<B: Backend> {
    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity - returns a tensor of the component label of each pixel.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components(img: BoolTensor<B>, connectivity: Connectivity) -> IntTensor<B> {
        cpu::connected_components::<B>(img, connectivity)
    }

    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity and collects statistics on each component - returns a tensor of the component
    /// label of each pixel, along with stats collected for each component.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components_with_stats(
        img: BoolTensor<B>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<B>, ConnectedStatsPrimitive<B>) {
        cpu::connected_components_with_stats(img, connectivity, opts)
    }
}
