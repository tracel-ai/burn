use burn_core::backend::Dispatch;
use burn_core::tensor::{Bool, DType, Float, Int, Tensor};

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

impl ConnectedComponents for Tensor<2, Bool> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<2, Int> {
        let settings = self.device().settings();
        Tensor::from_primitive(<Dispatch as BoolVisionOps>::connected_components(
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
            area: Tensor::from_primitive(area),
            left: Tensor::from_primitive(left),
            top: Tensor::from_primitive(top),
            right: Tensor::from_primitive(right),
            bottom: Tensor::from_primitive(bottom),
            max_label: Tensor::from_primitive(max_label),
        };
        (Tensor::from_primitive(labels), stats)
    }
}

impl Morphology for Tensor<3, Float> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        if matches!(self.dtype(), DType::QFloat(_)) {
            unimplemented!("Quantized float is not supported");
        }

        let out = <Dispatch as FloatVisionOps>::float_erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        );
        Tensor::from_primitive(out)
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        if matches!(self.dtype(), DType::QFloat(_)) {
            unimplemented!("Quantized float is not supported");
        }

        let out = <Dispatch as FloatVisionOps>::float_dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        );
        Tensor::from_primitive(out)
    }
}

impl Morphology for Tensor<3, Int> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_primitive(<Dispatch as IntVisionOps>::int_erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_primitive(<Dispatch as IntVisionOps>::int_dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }
}

impl Morphology for Tensor<3, Bool> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_primitive(<Dispatch as BoolVisionOps>::bool_erode(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_primitive(<Dispatch as BoolVisionOps>::bool_dilate(
            self.into_primitive(),
            kernel.into_primitive(),
            opts,
        ))
    }
}

impl Nms for Tensor<2> {
    fn nms(self, scores: Tensor<1>, options: NmsOptions) -> Tensor<1, Int> {
        if matches!(self.dtype(), DType::QFloat(_)) {
            unimplemented!("Quantized float is not supported");
        }

        let settings = self.device().settings();

        Tensor::from_primitive(<Dispatch as FloatVisionOps>::nms(
            self.into_primitive(),
            scores.into_primitive(),
            options,
            settings.int_dtype,
        ))
    }
}
