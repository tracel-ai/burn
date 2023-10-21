use crate::{
    element::WgpuElement,
    kernel::{unary_scalar, unary_scalar_inplace_default, WORKGROUP_DEFAULT},
    tensor::WgpuTensor,
    unary_scalar, unary_scalar_inplace,
};

pub(crate) fn clamp_min<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    value: E,
) -> WgpuTensor<E, D> {
    unary_scalar!(ClampMin, func "max");
    unary_scalar_inplace!(ClampMinInplace, func "max");

    if input.can_mut() {
        return unary_scalar_inplace_default::<ClampMinInplace, E, D>(input, value);
    }

    unary_scalar::<ClampMin, E, D, WORKGROUP_DEFAULT>(input, value)
}

pub(crate) fn clamp_max<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    value: E,
) -> WgpuTensor<E, D> {
    unary_scalar!(ClampMax, func "min");
    unary_scalar_inplace!(ClampMaxInPlace, func "min");

    if input.can_mut() {
        return unary_scalar_inplace_default::<ClampMaxInPlace, E, D>(input, value);
    }

    unary_scalar::<ClampMax, E, D, WORKGROUP_DEFAULT>(input, value)
}

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn clamp_min_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random([1, 5, 32, 32], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());

        let output = input.clamp_min(0.5);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp_min(0.5).into_data(), 3);
    }

    #[test]
    fn clamp_max_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random([1, 5, 32, 32], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());

        let output = input.clamp_max(0.5);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp_max(0.5).into_data(), 3);
    }
}
