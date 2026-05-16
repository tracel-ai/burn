use burn_core::tensor::{
    Bool, Float, Int, Tensor,
    backend::{TensorPrimitive, extension::Dispatch},
};

use crate::{
    BoolVisionOps, ConnectedStats, ConnectedStatsOptions, Connectivity, FloatVisionOps,
    IntVisionOps, MorphOptions, NmsOptions,
};

/// Connected components tensor extensions
pub trait ConnectedComponents {
    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity - returns a tensor of the component label of each pixel.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components(self, connectivity: Connectivity) -> Tensor<2, Int>;

    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity and collects statistics on each component - returns a tensor of the component
    /// label of each pixel, along with stats collected for each component.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<2, Int>, ConnectedStats);
}

/// Morphology tensor operations
pub trait Morphology {
    /// Erodes this tensor using the specified kernel.
    /// Assumes NHWC layout.
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self;
    /// Dilates this tensor using the specified kernel.
    /// Assumes NHWC layout.
    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self;
}

// /// Morphology tensor operations
// pub trait MorphologyKind: BasicOps<B> {
//     /// Erodes this tensor using the specified kernel
//     fn erode(
//         tensor: Self::Primitive,
//         kernel: BoolTensor<B>,
//         opts: MorphOptions<B, Self>,
//     ) -> Self::Primitive;
//     /// Dilates this tensor using the specified kernel
//     fn dilate(
//         tensor: Self::Primitive,
//         kernel: BoolTensor<B>,
//         opts: MorphOptions<B, Self>,
//     ) -> Self::Primitive;
// }

/// Non-maximum suppression tensor operations
pub trait Nms {
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
    fn nms(self, scores: Tensor<1, Float>, opts: NmsOptions) -> Tensor<1, Int>;
}

// THE IMPLEMENTATION WHICH SHOULD CALL DISPATCH
impl ConnectedComponents for Tensor<2, Bool> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<2, Int> {
        let settings = self.device().settings();
        Tensor::new(<Dispatch as BoolVisionOps>::connected_components(
            self.into_primitive(),
            connectivity,
            settings.int_dtype,
        ))
    }

    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<2, Int>, ConnectedStats) {
        let settings = self.device().settings();
        let (labels, area, left, top, right, bottom, max_label) =
            <Dispatch as BoolVisionOps>::connected_components_with_stats(
                self.into_primitive(),
                connectivity,
                options,
                settings.int_dtype,
            );
        let stats = ConnectedStats {
            area: Tensor::new(area),
            left: Tensor::new(left),
            top: Tensor::new(top),
            right: Tensor::new(right),
            bottom: Tensor::new(bottom),
            max_label: Tensor::new(max_label),
        };
        (Tensor::new(labels), stats)
    }
}

impl Morphology for Tensor<3, Float> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        let out = match self.into_primitive() {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(
                <Dispatch as FloatVisionOps>::float_erode(tensor, kernel.into_primitive(), opts),
            ),
            TensorPrimitive::QFloat(_) => unimplemented!(),
        };
        Tensor::new(out)
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        let out = match self.into_primitive() {
            TensorPrimitive::Float(tensor) => TensorPrimitive::Float(
                <Dispatch as FloatVisionOps>::float_dilate(tensor, kernel.into_primitive(), opts),
            ),
            TensorPrimitive::QFloat(_) => unimplemented!(),
        };
        Tensor::new(out)
    }
}

impl Morphology for Tensor<3, Int> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::new(<Dispatch as IntVisionOps>::int_erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::new(<Dispatch as IntVisionOps>::int_dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }
}

impl Morphology for Tensor<3, Bool> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::new(<Dispatch as BoolVisionOps>::bool_erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::new(<Dispatch as BoolVisionOps>::bool_dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }
}

impl Nms for Tensor<2> {
    fn nms(self, scores: Tensor<1>, options: NmsOptions) -> Tensor<1, Int> {
        let settings = self.device().settings();
        match (self.into_primitive(), scores.into_primitive()) {
            (TensorPrimitive::Float(boxes), TensorPrimitive::Float(scores)) => Tensor::new(
                <Dispatch as FloatVisionOps>::nms(boxes, scores, options, settings.int_dtype),
            ),
            _ => unimplemented!("Quantized inputs are not yet supported"),
        }
    }
}
