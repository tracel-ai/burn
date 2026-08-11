use crate::tensor::CubeTensor;
use crate::{CubeRuntime, ops::numeric::empty_device_dtype};
use alloc::{vec, vec::Vec};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };
    let (tensor, inverse_axes) = match scheme.store {
        cubecl::quant::scheme::QuantStore::PackedU32(dim)
        | cubecl::quant::scheme::QuantStore::PackedNative(dim)
            if dim != 0 =>
        {
            let rank = tensor.rank();
            let packed_axis = rank - dim - 1;
            let mut axes = (0..rank)
                .filter(|axis| *axis != packed_axis)
                .collect::<Vec<_>>();
            axes.push(packed_axis);

            let mut inverse_axes = vec![0; rank];
            for (axis, source_axis) in axes.iter().enumerate() {
                inverse_axes[*source_axis] = axis;
            }

            let tensor = (packed_axis..rank - 1).fold(tensor, |tensor, axis| {
                crate::ops::swap_dims(tensor, axis, axis + 1)
            });

            (tensor, Some(inverse_axes))
        }
        _ => (tensor, None),
    };
    let scheme = tensor.scheme();

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    cubek::quantization::dequantize::launch_ref(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        &scheme,
        dtype_to_storage_type(dtype),
    )
    .expect("Kernel to never fail");

    match inverse_axes {
        Some(axes) => crate::ops::permute(output, &axes),
        None => output,
    }
}
