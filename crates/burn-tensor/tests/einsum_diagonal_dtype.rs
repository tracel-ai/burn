//! A separate test binary isolates device defaults from other tensor tests.

use burn_tensor::{Device, IntDType, Tensor, einsum};

#[test]
fn diagonal_indices_do_not_use_the_devices_narrow_integer_default() {
    let mut device = Device::default();
    device.configure(IntDType::I8).unwrap();

    let data: [[f32; 12]; 12] =
        core::array::from_fn(|row| core::array::from_fn(|column| (row * 12 + column) as f32));
    let input = Tensor::<2>::from_floats(data, &device);
    let expected: [f32; 12] = core::array::from_fn(|index| (index * 13) as f32);

    // The final flattened diagonal index is 143, exceeding i8::MAX.
    let compiled = einsum!("ii->i", &input);
    let runtime = Tensor::<1>::einsum("ii->i", [input.into()]);
    assert_eq!(compiled.into_data().try_to_vec::<f32>().unwrap(), expected);
    assert_eq!(runtime.into_data().try_to_vec::<f32>().unwrap(), expected);
}
