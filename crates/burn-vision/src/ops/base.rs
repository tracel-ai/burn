use crate::{
    Point,
    backends::cpu::{self, MorphOp, morph},
};
use bon::Builder;
use burn_tensor::{
    Bool, Float, Int, Tensor, TensorKind, TensorPrimitive,
    backend::Backend,
    ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor},
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
pub struct MorphOptions<B: Backend, K: TensorKind<B>> {
    /// Anchor position within the kernel. Defaults to the center.
    pub anchor: Option<Point>,
    /// Number of iterations to apply
    #[builder(default = 1)]
    pub iterations: usize,
    /// Border type. Default: constant based on operation
    #[builder(default)]
    pub border_type: BorderType,
    /// Value of each channel for constant border type
    pub border_value: Option<Tensor<B, 1, K>>,
}

impl<B: Backend, K: TensorKind<B>> Default for MorphOptions<B, K> {
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

/// Vision capable backend, implemented by each backend
pub trait VisionBackend:
    BoolVisionOps + IntVisionOps + FloatVisionOps + QVisionOps + Backend
{
}

/// Vision ops on bool tensors
pub trait BoolVisionOps: Backend {
    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity - returns a tensor of the component label of each pixel.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components(img: BoolTensor<Self>, connectivity: Connectivity) -> IntTensor<Self> {
        cpu::connected_components::<Self>(img, connectivity)
    }

    /// Computes the connected components labeled image of boolean image with 4 or 8 way
    /// connectivity and collects statistics on each component - returns a tensor of the component
    /// label of each pixel, along with stats collected for each component.
    ///
    /// `img`- The boolean image tensor in the format [batches, height, width]
    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        cpu::connected_components_with_stats(img, connectivity, opts)
    }

    /// Erodes an input tensor with the specified kernel.
    fn bool_erode(
        input: BoolTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Bool>,
    ) -> BoolTensor<Self> {
        let input = Tensor::<Self, 3, Bool>::from_primitive(input);
        morph(input, kernel, MorphOp::Erode, opts).into_primitive()
    }

    /// Dilates an input tensor with the specified kernel.
    fn bool_dilate(
        input: BoolTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Bool>,
    ) -> BoolTensor<Self> {
        let input = Tensor::<Self, 3, Bool>::from_primitive(input);
        morph(input, kernel, MorphOp::Dilate, opts).into_primitive()
    }
}

/// Vision ops on int tensors
pub trait IntVisionOps: Backend {
    /// Erodes an input tensor with the specified kernel.
    fn int_erode(
        input: IntTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Int>,
    ) -> IntTensor<Self> {
        let input = Tensor::<Self, 3, Int>::from_primitive(input);
        morph(input, kernel, MorphOp::Erode, opts).into_primitive()
    }

    /// Dilates an input tensor with the specified kernel.
    fn int_dilate(
        input: IntTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Int>,
    ) -> IntTensor<Self> {
        let input = Tensor::<Self, 3, Int>::from_primitive(input);
        morph(input, kernel, MorphOp::Dilate, opts).into_primitive()
    }
}

/// Vision ops on float tensors
pub trait FloatVisionOps: Backend {
    /// Erodes an input tensor with the specified kernel.
    fn float_erode(
        input: FloatTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Float>,
    ) -> FloatTensor<Self> {
        let input = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(input));

        morph(input, kernel, MorphOp::Erode, opts)
            .into_primitive()
            .tensor()
    }

    /// Dilates an input tensor with the specified kernel.
    fn float_dilate(
        input: FloatTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Float>,
    ) -> FloatTensor<Self> {
        let input = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(input));
        morph(input, kernel, MorphOp::Dilate, opts)
            .into_primitive()
            .tensor()
    }
}

/// Vision ops on quantized float tensors
pub trait QVisionOps: Backend {
    /// Erodes an input tensor with the specified kernel.
    fn q_erode(
        input: QuantizedTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Float>,
    ) -> QuantizedTensor<Self> {
        let input = Tensor::<Self, 3>::from_primitive(TensorPrimitive::QFloat(input));
        match morph(input, kernel, MorphOp::Erode, opts).into_primitive() {
            TensorPrimitive::QFloat(tensor) => tensor,
            _ => unreachable!(),
        }
    }

    /// Dilates an input tensor with the specified kernel.
    fn q_dilate(
        input: QuantizedTensor<Self>,
        kernel: BoolTensor<Self>,
        opts: MorphOptions<Self, Float>,
    ) -> QuantizedTensor<Self> {
        let input = Tensor::<Self, 3>::from_primitive(TensorPrimitive::QFloat(input));
        match morph(input, kernel, MorphOp::Dilate, opts).into_primitive() {
            TensorPrimitive::QFloat(tensor) => tensor,
            _ => unreachable!(),
        }
    }
}
