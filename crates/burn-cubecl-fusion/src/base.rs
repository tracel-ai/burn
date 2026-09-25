use burn_fusion::stream::Context;
use burn_std::{
    DType, Metadata, Shape, Strides,
    quantization::{QParamTensor, global_scale_dtype},
    strides,
};
use cubecl::quant::scheme::{QuantScheme, ScaleDtype};
use cubecl::{
    client::Client,
    ir::AddressType,
    prelude::{TensorArg, TensorBinding},
    zspace::Tiling,
};

/// Defines a fallback operation when fusion isn't possible.
pub trait FallbackOperation: Send + Sync {
    /// Executes the fallback procedure.
    fn run(&self, context: &mut Context<CubeFusionHandle>);
}

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_std::quantization::QParams<QParamTensor>;

/// Handle to be used when fusing operations.
pub struct CubeFusionHandle {
    /// Compute client for jit.
    pub client: Client,
    /// The buffer where the data are stored.
    pub handle: cubecl::server::Handle,
    /// The device of the current tensor.
    pub device: cubecl::Device,
    /// The element type of the tensor.
    pub dtype: DType,
    /// The strides of the tensor.
    pub strides: Strides,
    /// How the tensor is stored: `None` for rows, under `strides` over the shape the IR states;
    /// otherwise storage tiles, whose physical dims the IR cannot state (it knows the logical
    /// shape only), so the handle carries them. A fused kernel reads rows and refuses a tiled
    /// input.
    pub tiles: Option<Tiles>,
    /// Quantization runtime parameters, if applicable
    pub qparams: Option<QParams>,
}

/// A storage-tiled buffer's physical shape and the tiling that folds it back into the logical
/// shape the IR states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tiles {
    /// The buffer's own dims, the fragments of the logical ones.
    pub shape: Shape,
    /// How many fragments each logical dim is stored as.
    pub tiling: Tiling,
}

impl core::fmt::Debug for CubeFusionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "CubeFusionHandle {{ device: {:?}, runtime: {}}}",
            self.device,
            self.client.name(),
        ))
    }
}

impl Clone for CubeFusionHandle {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
            tiles: self.tiles.clone(),
            dtype: self.dtype,
            qparams: self.qparams.clone(),
        }
    }
}

unsafe impl Send for CubeFusionHandle {}
unsafe impl Sync for CubeFusionHandle {}

impl CubeFusionHandle {
    /// Return the reference to a tensor handle, `shape` being the logical shape the IR states.
    pub fn binding(self, shape: Shape) -> TensorBinding {
        let (shape, tiling) = (self.physical_shape(shape), self.tiling());
        TensorBinding {
            handle: self.handle.binding(),
            strides: self.strides,
            shape,
            tiling,
        }
    }

    /// The buffer's own dims, `logical` being the shape the IR states: that shape for rows, the
    /// fragments for storage tiles.
    pub fn physical_shape(&self, logical: Shape) -> Shape {
        match &self.tiles {
            Some(tiles) => tiles.shape.clone(),
            None => logical,
        }
    }

    /// The buffer's metadata, `logical` being the shape the IR states: rows under that shape, or
    /// the storage tiles the handle carries, folded back to it by their tiling.
    pub fn metadata(&self, logical: Shape) -> Metadata {
        match &self.tiles {
            Some(tiles) => Metadata::new(tiles.shape.clone(), self.strides.clone())
                .with_tiling(tiles.tiling)
                .expect("a fusion handle's tiling describes its own rank"),
            None => Metadata::new(logical, self.strides.clone()),
        }
    }

    /// How many fragments each logical dim is stored as; untiled for rows.
    pub fn tiling(&self) -> Tiling {
        self.tiles
            .as_ref()
            .map(|tiles| tiles.tiling)
            .unwrap_or(Tiling::UNTILED)
    }

    pub fn required_address_type(&self) -> AddressType {
        match self.dtype {
            DType::QFloat(scheme) => {
                let len = self.handle.size() as usize * 8 / scheme.size_bits_value();
                AddressType::from_len(len)
            }
            _ => AddressType::from_len(self.handle.size() as usize / self.dtype.size()),
        }
    }

    /// Return the reference to a tensor argument.
    pub fn into_tensor_arg(self, shape: Shape) -> TensorArg {
        let handle = self.binding(shape);
        handle.into_tensor_arg()
    }

    /// Construct a separate tensor for the quantization scales, if present
    pub fn params(&self, scheme: QuantScheme) -> Option<Self> {
        let qparams = self.qparams.as_ref()?;
        // Only the block scale is threaded through below; a two-level scheme's per-tensor scale
        // would be silently dropped, so refuse rather than build a handle short one factor.
        assert!(
            global_scale_dtype(&scheme).is_none(),
            "fused kernels don't yet support a two-level scheme's per-tensor scale"
        );
        let mut handle = self.handle.clone();
        handle.offset_start = Some(qparams.scales.offset_start as u64);
        handle.offset_end = Some(qparams.scales.offset_end as u64);

        Some(Self {
            client: self.client.clone(),
            handle,
            device: self.device.clone(),
            dtype: match scheme.scale_dtype() {
                ScaleDtype::F32 => DType::F32,
                ScaleDtype::F16 => DType::F16,
                ScaleDtype::BF16 => DType::BF16,
                ScaleDtype::UE8M0 | ScaleDtype::UE4M3 => unimplemented!("Not yet supported"),
            },
            strides: qparams.scales.metadata.strides().clone(),
            tiles: None,
            qparams: None,
        })
    }
}

pub(crate) fn strides_dyn_rank(shape: &[usize]) -> Strides {
    let mut strides = strides![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}
