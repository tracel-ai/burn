use burn_tensor::{
    backend::Backend, ops::BoolTensor, BasicOps, Bool, Float, Int, Tensor, TensorPrimitive,
};

use crate::{BoolVisionOps, ConnectedStats, ConnectedStatsOptions, Connectivity, VisionBackend};

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

/// Morphology tensor operations
pub trait Morphology<B: Backend> {
    /// Erodes this tensor using the specified kernel
    fn erode(self, kernel: Tensor<B, 4, Bool>) -> Self;
    /// Dilates this tensor using the specified kernel
    fn dilate(self, kernel: Tensor<B, 4, Bool>) -> Self;
}

/// Morphology tensor operations
pub trait MorphologyKind<B: Backend>: BasicOps<B> {
    /// Erodes this tensor using the specified kernel
    fn erode(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive;
    /// Dilates this tensor using the specified kernel
    fn dilate(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive;
}

impl<B: BoolVisionOps> ConnectedComponents<B> for Tensor<B, 2, Bool> {
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

impl<B: VisionBackend, K: MorphologyKind<B>> Morphology<B> for Tensor<B, 4, K> {
    fn erode(self, kernel: Tensor<B, 4, Bool>) -> Self {
        Tensor::new(K::erode(self.into_primitive(), kernel.into_primitive()))
    }

    fn dilate(self, kernel: Tensor<B, 4, Bool>) -> Self {
        Tensor::new(K::dilate(self.into_primitive(), kernel.into_primitive()))
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Float {
    fn erode(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_erode(tensor, kernel))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_erode(tensor, kernel)),
        }
    }

    fn dilate(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_dilate(tensor, kernel))
            }
            TensorPrimitive::QFloat(tensor) => TensorPrimitive::QFloat(B::q_dilate(tensor, kernel)),
        }
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Int {
    fn erode(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        B::int_erode(tensor, kernel)
    }

    fn dilate(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        B::int_dilate(tensor, kernel)
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Bool {
    fn erode(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        B::bool_erode(tensor, kernel)
    }

    fn dilate(tensor: Self::Primitive, kernel: BoolTensor<B>) -> Self::Primitive {
        B::bool_dilate(tensor, kernel)
    }
}
