use super::*;
use burn_cubecl::kernel::{MaskFillStrategy, mask_fill};
use burn_tensor::Tolerance;
use burn_tensor::{Bool, Distribution, Element, Tensor, TensorPrimitive, backend::Backend};
use cubecl::prelude::InputScalar;

#[test]
fn mask_fill_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();
    let dtype_bool = <<TestBackend as Backend>::BoolElem as Element>::dtype();
    let dtype_ft = <FloatElem as Element>::dtype();

    let actual = Tensor::<TestBackend, 3>::from_primitive(TensorPrimitive::Float(mask_fill(
        tensor.into_primitive().tensor(),
        mask.into_primitive(),
        InputScalar::new(4.0, dtype_ft),
        MaskFillStrategy::Readonly,
        dtype_bool,
    )));
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn mask_fill_inplace_should_match_reference_backend() {
    let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();
    let dtype_bool = <<TestBackend as Backend>::BoolElem as Element>::dtype();
    let dtype_ft = <FloatElem as Element>::dtype();

    let actual = Tensor::<TestBackend, 3>::from_primitive(TensorPrimitive::Float(mask_fill::<_>(
        tensor.into_primitive().tensor(),
        mask.into_primitive(),
        InputScalar::new(4.0, dtype_ft),
        MaskFillStrategy::Inplace,
        dtype_bool,
    )));
    let expected = tensor_ref.mask_fill(mask_ref, 4.0);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[allow(clippy::type_complexity)]
fn inputs_mask_fill() -> (
    Tensor<TestBackend, 3>,
    Tensor<TestBackend, 3, Bool>,
    Tensor<ReferenceBackend, 3>,
    Tensor<ReferenceBackend, 3, Bool>,
) {
    let test_device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default, &test_device);
    let mask =
        Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0., 1.), &test_device)
            .lower_equal_elem(0.5);
    let ref_device = Default::default();
    let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data(), &ref_device);
    let mask_ref = Tensor::<ReferenceBackend, 3, Bool>::from_data(mask.to_data(), &ref_device);

    (tensor, mask, tensor_ref, mask_ref)
}
