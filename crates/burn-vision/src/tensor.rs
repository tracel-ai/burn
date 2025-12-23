use burn_tensor::{
    BasicOps, Bool, Float, Int, Tensor, TensorKind, TensorPrimitive, backend::Backend,
    ops::BoolTensor,
};

use crate::{
    BoolVisionOps, ConnectedStats, ConnectedStatsOptions, Connectivity, MorphOptions, NmsOptions,
    VisionBackend,
};

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
pub trait Morphology<B: Backend, K: TensorKind<B>> {
    /// Erodes this tensor using the specified kernel.
    /// Assumes NHWC layout.
    fn erode(self, kernel: Tensor<B, 2, Bool>, opts: MorphOptions<B, K>) -> Self;
    /// Dilates this tensor using the specified kernel.
    /// Assumes NHWC layout.
    fn dilate(self, kernel: Tensor<B, 2, Bool>, opts: MorphOptions<B, K>) -> Self;
}

/// Morphology tensor operations
pub trait MorphologyKind<B: Backend>: BasicOps<B> {
    /// Erodes this tensor using the specified kernel
    fn erode(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive;
    /// Dilates this tensor using the specified kernel
    fn dilate(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive;
}

/// Non-maximum suppression tensor operations
pub trait Nms<B: Backend> {
    /// Perform Non-Maximum Suppression on this tensor of bounding boxes.
    ///
    /// Returns indices of kept boxes after suppressing overlapping detections.
    /// Boxes are processed in descending score order; a box suppresses all
    /// lower-scoring boxes with IoU > threshold.
    ///
    /// # Arguments
    /// * `self` - Bounding boxes as \[N, 4\] tensor in (x1, y1, x2, y2) format
    /// * `scores` - Confidence scores as \[N\] tensor
    /// * `options` - NMS options (IoU threshold, score threshold, max boxes)
    ///
    /// # Returns
    /// Indices of kept boxes as \[M\] tensor where M <= N
    fn nms(self, scores: Tensor<B, 1, Float>, opts: NmsOptions) -> Tensor<B, 1, Int>;
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

impl<B: VisionBackend, K: MorphologyKind<B>> Morphology<B, K> for Tensor<B, 3, K> {
    fn erode(self, kernel: Tensor<B, 2, Bool>, opts: MorphOptions<B, K>) -> Self {
        Tensor::new(K::erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<B, 2, Bool>, opts: MorphOptions<B, K>) -> Self {
        Tensor::new(K::dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Float {
    fn erode(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_erode(tensor, kernel, opts))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_erode(tensor, kernel, opts))
            }
        }
    }

    fn dilate(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                TensorPrimitive::Float(B::float_dilate(tensor, kernel, opts))
            }
            TensorPrimitive::QFloat(tensor) => {
                TensorPrimitive::QFloat(B::q_dilate(tensor, kernel, opts))
            }
        }
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Int {
    fn erode(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        B::int_erode(tensor, kernel, opts)
    }

    fn dilate(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        B::int_dilate(tensor, kernel, opts)
    }
}

impl<B: VisionBackend> MorphologyKind<B> for Bool {
    fn erode(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        B::bool_erode(tensor, kernel, opts)
    }

    fn dilate(
        tensor: Self::Primitive,
        kernel: BoolTensor<B>,
        opts: MorphOptions<B, Self>,
    ) -> Self::Primitive {
        B::bool_dilate(tensor, kernel, opts)
    }
}

impl<B: VisionBackend> Nms<B> for Tensor<B, 2> {
    fn nms(self, scores: Tensor<B, 1>, options: NmsOptions) -> Tensor<B, 1, Int> {
        match (self.into_primitive(), scores.into_primitive()) {
            (TensorPrimitive::Float(boxes), TensorPrimitive::Float(scores)) => {
                Tensor::<B, 1, Int>::from_primitive(B::nms(boxes, scores, options))
            }
            _ => todo!("Quantized inputs are not yet supported"),
        }
    }
}
