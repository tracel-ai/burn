use crate::{element::WgpuElement, tensor::WgpuTensor};
use burn_tensor::Shape;

pub(crate) fn shape_out<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
) -> Shape<D> {
    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });
    shape_out[D - 2] = lhs.shape.dims[D - 2];
    shape_out[D - 1] = rhs.shape.dims[D - 1];
    Shape::new(shape_out)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::tensor::WgpuTensor;
    use crate::tests::{ReferenceTensor, TestTensor};
    use burn_tensor::Shape;

    pub(crate) fn same_as_reference<F, const D: usize, S>(func: F, shape_lhs: S, shape_rhs: S)
    where
        F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));

        let x_wgpu = TestTensor::from_data(x.to_data());
        let y_wgpu = TestTensor::from_data(y.to_data());

        let z_reference = x.matmul(y);

        let z = func(x_wgpu.into_primitive(), y_wgpu.into_primitive());
        let z = TestTensor::from_primitive(z);

        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }

    pub(crate) fn same_as_reference_swapped_dims<F, const D: usize, S>(
        func: F,
        swap_lhs: [usize; 2],
        swap_rhs: [usize; 2],
        shape_lhs: S,
        shape_rhs: S,
    ) where
        F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));

        let x_wgpu = TestTensor::from_data(x.to_data());
        let y_wgpu = TestTensor::from_data(y.to_data());

        let z_reference = x
            .swap_dims(swap_lhs[0], swap_lhs[1])
            .matmul(y.swap_dims(swap_rhs[0], swap_rhs[1]));

        let z = func(
            x_wgpu.swap_dims(swap_lhs[0], swap_lhs[1]).into_primitive(),
            y_wgpu.swap_dims(swap_rhs[0], swap_rhs[1]).into_primitive(),
        );
        let z = TestTensor::from_primitive(z);

        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }
}
