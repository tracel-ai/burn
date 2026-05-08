use crate::{
    Point,
    backends::cpu::{self, MorphOp, morph},
};
use bon::Builder;

use burn_core as burn; // for backend_extension
use burn_core::tensor::{
    Int, IntDType, Scalar, Tensor,
    backend::{Backend, extension::backend_extension},
    ops::IntTensor,
    read_sync,
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

/// Options for morphology ops
#[derive(Clone, Debug, Builder)]
pub struct MorphOptions {
    /// Anchor position within the kernel. Defaults to the center.
    pub anchor: Option<Point>,
    /// Number of iterations to apply
    #[builder(default = 1)]
    pub iterations: usize,
    /// Border type. Default: constant based on operation
    #[builder(default)]
    pub border_type: BorderType,
    /// Value of each channel for constant border type
    pub border_value: Option<Vec<Scalar>>,
}

impl Default for MorphOptions {
    fn default() -> Self {
        Self {
            anchor: Default::default(),
            iterations: 1,
            border_type: Default::default(),
            border_value: Default::default(),
        }
    }
}

/// Morphology border type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum BorderType {
    /// Constant border with per-channel value. If no value is provided, the value is picked based
    /// on the morph op.
    #[default]
    Constant,
    /// Replicate first/last element
    Replicate,
    /// Reflect start/end elements
    Reflect,
    /// Reflect start/end elements, ignoring the first/last element
    Reflect101,
    /// Not supported for erode/dilate
    Wrap,
}

/// Stats collected by the connected components analysis
///
/// Disabled analyses may be aliased to labels
#[derive(Clone, Debug)]
pub struct ConnectedStats {
    /// Total area of each component
    pub area: Tensor<1, Int>,
    /// Topmost y coordinate in the component
    pub top: Tensor<1, Int>,
    /// Leftmost x coordinate in the component
    pub left: Tensor<1, Int>,
    /// Rightmost x coordinate in the component
    pub right: Tensor<1, Int>,
    /// Bottommost y coordinate in the component
    pub bottom: Tensor<1, Int>,
    /// Scalar tensor of the max label
    pub max_label: Tensor<1, Int>,
}

/// Primitive version of [`ConnectedStats`], to be returned by the backend
pub type ConnectedStatsPrimitive<B> = (
    // Total area of each component
    IntTensor<B>,
    // Leftmost x coordinate in the component
    IntTensor<B>,
    // Topmost y coordinate in the component
    IntTensor<B>,
    // Rightmost x coordinate in the component
    IntTensor<B>,
    // Bottommost y coordinate in the component
    IntTensor<B>,
    // Scalar tensor of the max label
    IntTensor<B>,
);

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

/// Non-Maximum Suppression options.
#[derive(Clone, Copy, Debug)]
pub struct NmsOptions {
    /// IoU threshold for suppression (default: 0.5).
    /// Boxes with IoU > threshold with a higher-scoring box are suppressed.
    pub iou_threshold: f32,
    /// Score threshold to filter boxes before NMS (default: 0.0, i.e., no filtering).
    /// Boxes with score < score_threshold are discarded.
    pub score_threshold: f32,
    /// Maximum number of boxes to keep (0 = unlimited).
    pub max_output_boxes: usize,
}

impl Default for NmsOptions {
    fn default() -> Self {
        Self {
            iou_threshold: 0.5,
            score_threshold: 0.0,
            max_output_boxes: 0,
        }
    }
}

#[cfg(feature = "flex")]
use burn_flex::Flex;

#[cfg(feature = "wgpu")]
use burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

/// Vision capable backend, implemented by each backend
#[backend_extension(
    Flex: cfg(feature = "flex"),
    Wgpu: cfg(feature = "wgpu"),
    Cuda: cfg(feature = "cuda"),
)]
pub trait VisionBackend: Backend + BoolVisionOps + IntVisionOps + FloatVisionOps {}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Wgpu: cfg(feature = "wgpu"),
    Cuda: cfg(feature = "cuda"),
)]
/// Vision ops on bool tensors
pub trait BoolVisionOps: Backend {
    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity - returns a tensor of the component label of each pixel.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        let device = Self::bool_device(&img);
        Self::int_from_data(
            cpu::connected_components::<Self>(img, connectivity, out_dtype),
            &device,
        )
    }

    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity and collects statistics on each component - returns a tensor of the component
    /// label of each pixel, along with stats collected for each component.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    // TODO: support struct return types that encapsulate tensors with `#[backend_extension]`
    #[allow(clippy::type_complexity)]
    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
        out_dtype: IntDType,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
        IntTensor<Self>,
    ) {
        let device = Self::bool_device(&img);
        let (labels, (area, top, left, right, bottom, max_label)) =
            cpu::connected_components_with_stats::<Self>(img, connectivity, opts, out_dtype);
        (
            Self::int_from_data(labels, &device),
            area,
            top,
            left,
            right,
            bottom,
            max_label,
        )
    }

    /// Erodes an input tensor with the specified kernel.
    fn bool_erode(
        input: BoolTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> BoolTensor<Self> {
        let device = Self::bool_device(&input);
        let input = read_sync(Self::bool_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::bool_from_data(morph::<Self>(input, kernel, MorphOp::Erode, opts), &device)
    }

    /// Dilates an input tensor with the specified kernel.
    fn bool_dilate(
        input: BoolTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> BoolTensor<Self> {
        let device = Self::bool_device(&input);
        let input = read_sync(Self::bool_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::bool_from_data(morph::<Self>(input, kernel, MorphOp::Dilate, opts), &device)
    }
}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Wgpu: cfg(feature = "wgpu"),
    Cuda: cfg(feature = "cuda"),
)]
/// Vision ops on int tensors
pub trait IntVisionOps: Backend {
    /// Erodes an input tensor with the specified kernel.
    fn int_erode(
        input: IntTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> IntTensor<Self> {
        let device = Self::int_device(&input);
        let input = read_sync(Self::int_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::int_from_data(morph::<Self>(input, kernel, MorphOp::Erode, opts), &device)
    }

    /// Dilates an input tensor with the specified kernel.
    fn int_dilate(
        input: IntTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> IntTensor<Self> {
        let device = Self::int_device(&input);
        let input = read_sync(Self::int_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::int_from_data(morph::<Self>(input, kernel, MorphOp::Dilate, opts), &device)
    }
}

#[backend_extension(
    Flex: cfg(feature = "flex"),
    Wgpu: cfg(feature = "wgpu"),
    Cuda: cfg(feature = "cuda"),
)]
/// Vision ops on float tensors
pub trait FloatVisionOps: Backend {
    /// Erodes an input tensor with the specified kernel.
    fn float_erode(
        input: FloatTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> FloatTensor<Self> {
        let device = Self::float_device(&input);
        let input = read_sync(Self::float_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::float_from_data(morph::<Self>(input, kernel, MorphOp::Erode, opts), &device)
    }

    /// Dilates an input tensor with the specified kernel.
    fn float_dilate(
        input: FloatTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions,
    ) -> FloatTensor<Self> {
        let device = Self::float_device(&input);
        let input = read_sync(Self::float_into_data(input)).expect("Should read data");
        let kernel = read_sync(Self::bool_into_data(kernel)).expect("Should read data");

        Self::float_from_data(morph::<Self>(input, kernel, MorphOp::Dilate, opts), &device)
    }

    /// Perform Non-Maximum Suppression on bounding boxes.
    ///
    /// Returns indices of kept boxes after suppressing overlapping detections.
    /// Boxes are processed in descending score order; a box suppresses all
    /// lower-scoring boxes with IoU > threshold.
    ///
    /// # Arguments
    /// * `boxes` - Bounding boxes as \[N, 4\] tensor in (x1, y1, x2, y2) format
    /// * `scores` - Confidence scores as \[N\] tensor
    /// * `options` - NMS options (IoU threshold, score threshold, max boxes)
    ///
    /// # Returns
    /// Indices of kept boxes as \[M\] tensor where M <= N
    fn nms(
        boxes: FloatTensor<Self>,
        scores: FloatTensor<Self>,
        options: NmsOptions,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        let device = Self::float_device(&boxes);
        let boxes = read_sync(Self::float_into_data(boxes)).expect("Should read data");
        let scores = read_sync(Self::float_into_data(scores)).expect("Should read data");

        match cpu::nms(boxes, scores, options, out_dtype) {
            Some(data) => Self::int_from_data(data, &device),
            None => Self::int_zeros([0].into(), &device, out_dtype),
        }
    }
}
