use burn_backend::{DType, Shape, TensorMetadata as _, quantization::QParamTensor};
use burn_std::{Metadata, Strides};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::{client::ComputeClient, server::Handle};

use crate::CubeRuntime;

use super::CubeTensor;

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_backend::quantization::QParams<QParamTensor>;

impl<R: CubeRuntime> CubeTensor<R> {
    /// Create a new quantized tensor
    pub fn new_quantized(
        client: ComputeClient<R>,
        handle: Handle,
        shape: Shape,
        device: R::Device,
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
    pub fn quantized_handles(&self) -> Option<(CubeTensor<R>, CubeTensor<R>)> {
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

    /// Construct a separate tensor for the quantization scales, if present
    pub fn scales(&self) -> Option<CubeTensor<R>> {
        let qparams = self.qparams.as_ref()?;
        let mut handle = self.handle.clone();
        handle.offset_start = Some(qparams.scales.offset_start as u64);
        handle.offset_end = Some(qparams.scales.offset_end as u64);

        Some(CubeTensor::new(
            self.client.clone(),
            handle,
            qparams.scales.metadata.clone(),
            self.device.clone(),
            qparams.scales.dtype,
        ))
    }
}
