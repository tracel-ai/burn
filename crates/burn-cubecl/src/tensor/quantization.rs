use crate::CubeDevice;
use burn_backend::{DType, Shape, TensorMetadata as _, quantization::QParamTensor};
use burn_std::{Metadata, Strides};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::{client::Client, server::Handle};

use super::CubeTensor;

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_backend::quantization::QParams<QParamTensor>;

impl CubeTensor {
    /// Create a new quantized tensor
    pub fn new_quantized(
        client: Client,
        handle: Handle,
        shape: Shape,
        device: CubeDevice,
        strides: Strides,
        dtype: DType,
        qparams: QParams,
    ) -> Self {
        CubeTensor {
            client,
            handle,
            meta: Box::new(Metadata::new(shape, strides)),
            device,
            dtype,
            qparams: Some(qparams),
        }
    }

    /// Returns the two tensors: (values, params) for a quantized tensor.
    /// For the values, native types that aren't supported as a normal `DType` will be returned
    /// as an unsigned integer tensor representing the bits. Should be reconstructed using `from_bits`
    /// in kernels.
    ///
    /// A storage-tiled tensor's values keep its metadata whole: its physical dims already state
    /// the packing, and only a kernel that reads tiles may bind them — any other refuses them
    /// as rows. The scales keep their own metadata, tiled or not.
    pub fn quantized_handles(&self) -> Option<(CubeTensor, CubeTensor)> {
        let params = self.scales()?;
        let scheme = match self.dtype {
            DType::QFloat(sc) => sc,
            _ => return None,
        };
        let values = match scheme.store {
            QuantStore::Native => match scheme.value {
                QuantValue::Q8F | QuantValue::Q8S => CubeTensor {
                    client: self.client.clone(),
                    handle: self.handle.clone(),
                    meta: self.meta.clone(),
                    device: self.device.clone(),
                    dtype: DType::I8,
                    qparams: None,
                },
                QuantValue::E4M3 | QuantValue::E5M2 => CubeTensor {
                    client: self.client.clone(),
                    handle: self.handle.clone(),
                    meta: self.meta.clone(),
                    device: self.device.clone(),
                    dtype: DType::U8,
                    qparams: None,
                },
                QuantValue::Q4F
                | QuantValue::Q4S
                | QuantValue::Q2F
                | QuantValue::Q2S
                | QuantValue::E2M1 => {
                    panic!("Can't store native sub-byte values")
                }
            },
            QuantStore::PackedU32(_) if self.meta.is_tiled() => self.stored_values(DType::U32),
            QuantStore::PackedNative(_) if self.meta.is_tiled() => self.stored_values(DType::U8),
            QuantStore::PackedU32(packed_dim) => {
                let packed_dim = self.rank() - packed_dim - 1;
                let mut shape = self.shape();
                shape[packed_dim] = shape[packed_dim].div_ceil(scheme.num_quants());

                CubeTensor {
                    client: self.client.clone(),
                    handle: self.handle.clone(),
                    meta: Box::new(Metadata::new(shape, self.meta.strides.clone())),
                    device: self.device.clone(),
                    dtype: DType::U32,
                    qparams: None,
                }
            }
            QuantStore::PackedNative(packed_dim) => match scheme.value {
                QuantValue::E2M1 => {
                    let packed_dim = self.rank() - packed_dim - 1;
                    let mut shape = self.shape();
                    shape[packed_dim] = shape[packed_dim].div_ceil(scheme.num_quants());

                    CubeTensor {
                        client: self.client.clone(),
                        handle: self.handle.clone(),
                        meta: Box::new(Metadata::new(shape, self.meta.strides.clone())),
                        device: self.device.clone(),
                        dtype: DType::U8,
                        qparams: None,
                    }
                }
                other => panic!("{other:?} doesn't support native packing"),
            },
        };

        Some((values, params))
    }

    /// The packed values of a storage-tiled tensor, as stored.
    fn stored_values(&self, dtype: DType) -> CubeTensor {
        CubeTensor {
            client: self.client.clone(),
            handle: self.handle.clone(),
            meta: self.meta.clone(),
            device: self.device.clone(),
            dtype,
            qparams: None,
        }
    }

    /// Construct a separate tensor for the quantization scales, if present
    pub fn scales(&self) -> Option<CubeTensor> {
        self.param_tensor(|qparams| Some(&qparams.scales))
    }

    /// Construct a separate tensor for the per-tensor scale, for a two-level scheme.
    pub fn global(&self) -> Option<CubeTensor> {
        self.param_tensor(|qparams| qparams.global.as_ref())
    }

    fn param_tensor(
        &self,
        select: impl Fn(&QParams) -> Option<&QParamTensor>,
    ) -> Option<CubeTensor> {
        let param = select(self.qparams.as_ref()?)?;
        let mut handle = self.handle.clone();
        handle.offset_start = Some(param.offset_start as u64);
        handle.offset_end = Some(param.offset_end as u64);

        Some(CubeTensor::new(
            self.client.clone(),
            handle,
            param.metadata.clone(),
            self.device.clone(),
            param.dtype,
        ))
    }
}

#[cfg(all(
    test,
    any(feature = "wgpu", feature = "cpu", feature = "cuda", feature = "hip")
))]
mod storage_tiled {
    use burn_backend::{
        DType, TensorMetadata,
        ops::QTensorOps,
        quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
    };
    use burn_std::{FloatDType, Metadata, TensorData};
    use cubecl::zspace::Tiling;

    use crate::{CubeBackend, CubeDevice, kernel::untile, ops::from_data, tensor::CubeTensor};

    /// A `[64, 128]` weight quantized four values to a word, and the same tensor stated as
    /// stored in 32x32 tiles: what a load leaves when it lays a weight out for a kernel that
    /// reads tiles. Only the metadata is restated; no test here reads the tiled bytes.
    fn quantized_and_tiled() -> (CubeTensor, CubeTensor) {
        let device = CubeDevice::default();
        let values = (0..64 * 128)
            .map(|i| i as f32 / 8192.0 - 0.5)
            .collect::<Vec<_>>();
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::PackedU32(0))
            .per_block([32], ScaleDtype::F32);
        let quantized = CubeBackend::quantize_dynamic(
            from_data(TensorData::new(values, [64, 128]), &device),
            &scheme,
        );
        let stored = Metadata::new([2, 4, 32, 32], [4 * 32 * 32, 32 * 32, 32, 1])
            .with_tiling(Tiling::new(&[2, 2]).expect("two fragments of each dim"))
            .expect("the tiling describes the four physical dims");
        let mut tiled = quantized.clone();
        tiled.meta = Box::new(stored);
        (quantized, tiled)
    }

    /// A tiled weight still presents its logical shape, and its values carry the tiles to the
    /// kernel that reads them, where an untiled weight's values are rows of words as before.
    #[test]
    fn a_tiled_weight_hands_its_tiles_to_its_values() {
        let (quantized, tiled) = quantized_and_tiled();
        assert_eq!(tiled.shape().as_slice(), &[64, 128]);

        let (values, scales) = tiled.quantized_handles().unwrap();
        assert_eq!(values.meta, tiled.meta);
        assert_eq!(values.dtype, DType::U32);
        assert_eq!(scales.meta, quantized.scales().unwrap().meta);

        let (rows, _) = quantized.quantized_handles().unwrap();
        assert!(!rows.meta.is_tiled());
        assert_eq!(rows.meta.shape().as_slice(), &[64, 32]);
    }

    /// Reading a tiled weight's values as rows would compute garbage without an error, so every
    /// path that would is refused by name.
    #[test]
    #[should_panic(expected = "dequantize: a storage-tiled quantized tensor")]
    fn a_tiled_weight_is_not_dequantized() {
        let (_, tiled) = quantized_and_tiled();
        CubeBackend::dequantize(tiled, FloatDType::F32);
    }

    #[test]
    #[should_panic(expected = "untile: a storage-tiled quantized tensor")]
    fn a_tiled_weight_is_not_untiled() {
        let (_, tiled) = quantized_and_tiled();
        untile(tiled);
    }

    /// A layout rewrite lays a tensor back first, so it is refused with the same reason.
    #[test]
    #[should_panic(expected = "untile: a storage-tiled quantized tensor")]
    fn a_tiled_weight_is_not_transposed() {
        let (_, tiled) = quantized_and_tiled();
        CubeBackend::q_swap_dims(tiled, 0, 1);
    }

    /// Tiles are laid out per machine at load; what is saved is the rows every machine reads.
    #[test]
    #[should_panic(expected = "q_into_data: a storage-tiled quantized tensor is not saved")]
    fn a_tiled_weight_is_not_saved() {
        let (_, tiled) = quantized_and_tiled();
        let _ = burn_std::future::block_on(CubeBackend::q_into_data(tiled));
    }
}
