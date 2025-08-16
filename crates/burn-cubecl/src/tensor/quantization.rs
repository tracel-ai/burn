use burn_tensor::{DType, Shape, quantization::QParamTensor};
use cubecl::{client::ComputeClient, server::Handle};

use crate::CubeRuntime;

use super::CubeTensor;

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_tensor::quantization::QParams<QParamTensor>;

impl<R: CubeRuntime> CubeTensor<R> {
    /// Create a new quantized tensor
    pub fn new_quantized(
        client: ComputeClient<R::Server, R::Channel>,
        handle: Handle,
        shape: Shape,
        device: R::Device,
        strides: Vec<usize>,
        dtype: DType,
        qparams: QParams,
    ) -> Self {
        CubeTensor {
            client,
            handle,
            shape,
            device,
            strides,
            dtype,
            qparams: Some(qparams),
        }
    }

    /// Returns the two tensors: (values, params) for a quantized tensor.
    pub fn quantized_handles(&self) -> Option<(CubeTensor<R>, CubeTensor<R>)> {
        let params = self.scales()?;
        let scheme = match self.dtype {
            DType::QFloat(sc) => sc,
            _ => return None,
        };
        let values = match scheme.store {
            cubecl_quant::scheme::QuantStore::Native => match scheme.value {
                cubecl_quant::scheme::QuantValue::QInt8 => CubeTensor {
                    client: self.client.clone(),
                    handle: self.handle.clone(),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                    strides: self.strides.clone(),
                    dtype: DType::I8,
                    qparams: None,
                },
            },
            cubecl_quant::scheme::QuantStore::U32 => {
                let rank = self.shape.num_dims();
                let mut shape = self.shape.clone();
                shape.dims[rank - 1] /= scheme.num_quants();

                CubeTensor {
                    client: self.client.clone(),
                    handle: self.handle.clone(),
                    shape,
                    device: self.device.clone(),
                    strides: self.strides.clone(),
                    dtype: DType::U32,
                    qparams: None,
                }
            }
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
            qparams.scales.shape.clone(),
            self.device.clone(),
            qparams.scales.strides.clone(),
            qparams.scales.dtype,
        ))
    }
}
