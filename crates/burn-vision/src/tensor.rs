use burn_core::backend::Dispatch;
use burn_core::tensor::{Bool, DType, Float, Int, Tensor};

use crate::{
    BoolVisionOps, ConnectedStats, ConnectedStatsOptions, Connectivity, FloatVisionOps,
    IntVisionOps, MorphOptions, NmsOptions,
};

/// Weights of each RGB channel for grayscale conversion, as per
/// [OpenCV](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html).
const RGB2GRAY_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];
const EPS: f32 = 1e-8;

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

/// Color space conversion tensor operations.
pub trait ColorConversion {
    /// Converts a batch of images from the RGB color space to grayscale.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`
    ///           in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The grayscale image tensor of shape `[batch, 1, height, width]`.
    fn rgb2gray(self) -> Tensor<4>;
    /// Converts a batch of grayscale images to the RGB color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, 1, height, width]`.
    ///
    /// # Returns
    /// The converted batched image tensor of shape `[batch, 3, height, width]`
    fn gray2rgb(self) -> Tensor<4>;
    /// Converts a batch of images from the RGB color space to the HSV color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`
    ///           in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The same-shape image tensor in HSV color space.
    fn rgb2hsv(self) -> Tensor<4>;
    /// Converts a batch of images from the HSV color space to the RGB color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`.
    ///           The first channel (hue) is in the `0.0..360.0` range.
    ///           The other two (saturation and value) are in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The same-shape image tensor in RGB color space.
    fn hsv2rgb(self) -> Tensor<4>;
}

impl ConnectedComponents for Tensor<2, Bool> {
    fn connected_components(self, connectivity: Connectivity) -> Tensor<2, Int> {
        let settings = self.device().settings();
        Tensor::from_dispatch(<Dispatch as BoolVisionOps>::connected_components(
            self.into_dispatch(),
            connectivity,
            settings.int_dtype,
        ))
    }

    fn connected_components_with_stats(
        self,
        connectivity: Connectivity,
        options: ConnectedStatsOptions,
    ) -> (Tensor<2, Int>, ConnectedStats) {
        println!("Tensor::connected_components_with_stats");
        let settings = self.device().settings();
        let (labels, stats) = <Dispatch as BoolVisionOps>::connected_components_with_stats(
            self.into_dispatch(),
            connectivity,
            options,
            settings.int_dtype,
        );

        let stats = ConnectedStats {
            area: Tensor::from_dispatch(stats.area),
            left: Tensor::from_dispatch(stats.left),
            top: Tensor::from_dispatch(stats.top),
            right: Tensor::from_dispatch(stats.right),
            bottom: Tensor::from_dispatch(stats.bottom),
            max_label: Tensor::from_dispatch(stats.max_label),
        };
        (Tensor::from_dispatch(labels), stats)
    }
}

impl Morphology for Tensor<3, Float> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        if matches!(self.dtype(), DType::QFloat(_)) {
            unimplemented!("Quantized float is not supported");
        }

        let out = <Dispatch as FloatVisionOps>::float_erode(
            self.into_dispatch(),
            kernel.into_dispatch(),
            opts,
        );
        Tensor::from_dispatch(out)
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        if matches!(self.dtype(), DType::QFloat(_)) {
            unimplemented!("Quantized float is not supported");
        }

        let out = <Dispatch as FloatVisionOps>::float_dilate(
            self.into_dispatch(),
            kernel.into_dispatch(),
            opts,
        );
        Tensor::from_dispatch(out)
    }
}

impl Morphology for Tensor<3, Int> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_dispatch(<Dispatch as IntVisionOps>::int_erode(
            self.into_dispatch(),
            kernel.into_dispatch(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_dispatch(<Dispatch as IntVisionOps>::int_dilate(
            self.into_dispatch(),
            kernel.into_dispatch(),
            opts,
        ))
    }
}

impl Morphology for Tensor<3, Bool> {
    fn erode(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_dispatch(<Dispatch as BoolVisionOps>::bool_erode(
            self.into_dispatch(),
            kernel.into_dispatch(),
            opts,
        ))
    }

    fn dilate(self, kernel: Tensor<2, Bool>, opts: MorphOptions) -> Self {
        Tensor::from_dispatch(<Dispatch as BoolVisionOps>::bool_dilate(
            self.into_dispatch(),
            kernel.into_dispatch(),
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

        Tensor::from_dispatch(<Dispatch as FloatVisionOps>::nms(
            self.into_dispatch(),
            scores.into_dispatch(),
            options,
            settings.int_dtype,
        ))
    }
}

fn channel(img: &Tensor<4>, at: usize) -> Tensor<4> {
    img.clone().narrow(1, at, 1)
}

/// One channel back out of a hue, its value and its chroma.
/// `target` - target channel : `0` for red, `1` for green and `2` for blue.
fn from_hue(sixths: &Tensor<4>, target: usize, value: &Tensor<4>, chroma: &Tensor<4>) -> Tensor<4> {
    // Rotate the wheel depending on the target channel.
    let rotated = sixths
        .clone()
        .add_scalar(5.0 - (target as f32) * 2.0)
        .remainder_scalar(6.0);
    let ramp = rotated.clone().min_pair(4.0 - rotated).clamp(0.0, 1.0);

    value.clone() - chroma.clone() * ramp
}

/// Forumlas derived from the [OpenCV docs](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html).
impl ColorConversion for Tensor<4, Float> {
    fn rgb2gray(self) -> Tensor<4> {
        channel(&self, 0).mul_scalar(RGB2GRAY_WEIGHTS[0])
            + channel(&self, 1).mul_scalar(RGB2GRAY_WEIGHTS[1])
            + channel(&self, 2).mul_scalar(RGB2GRAY_WEIGHTS[2])
    }

    fn gray2rgb(self) -> Tensor<4> {
        self.repeat_dim(1, 3)
    }

    fn rgb2hsv(self) -> Tensor<4> {
        let (red, green, blue) = (channel(&self, 0), channel(&self, 1), channel(&self, 2));
        let value = self.clone().max_dim(1);
        let chroma = value.clone() - self.min_dim(1);

        // Clamped values are always multiplied by 0, so this has no numerical impact.
        let by_chroma = chroma.clone().clamp_min(EPS);
        let by_value = value.clone().clamp_min(EPS);

        // OpenCV takes the first of the three that matches, so
        // red wins a tie with green and green one with blue.
        let sixths = (red.clone() - green.clone()) / by_chroma.clone() + 4.0;
        let sixths = sixths.mask_where(
            value.clone().equal(green.clone()),
            (blue.clone() - red.clone()) / by_chroma.clone() + 2.0,
        );
        let sixths = sixths.mask_where(value.clone().equal(red), (green - blue) / by_chroma);

        Tensor::cat(
            vec![
                sixths.mul_scalar(360.0 / 6.0).remainder_scalar(360.0),
                chroma / by_value,
                value,
            ],
            1,
        )
    }

    fn hsv2rgb(self) -> Tensor<4> {
        let (hue, saturation, value) = (channel(&self, 0), channel(&self, 1), channel(&self, 2));
        let sixths = hue.div_scalar(360.0 / 6.0);
        let chroma = value.clone() * saturation;

        Tensor::cat(
            vec![
                from_hue(&sixths, 0, &value, &chroma),
                from_hue(&sixths, 1, &value, &chroma),
                from_hue(&sixths, 2, &value, &chroma),
            ],
            1,
        )
    }
}
