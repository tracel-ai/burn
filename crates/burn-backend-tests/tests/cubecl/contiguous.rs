use super::*;
use burn_tensor::Tolerance;

#[test]
pub fn into_contiguous_match_reference_backend_1() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    for shape in [
        [4, 4, 4, 4],
        [32, 42, 24, 48],
        [8, 3, 7, 4],
        [1, 4, 1, 1],
        [1, 32, 256, 128],
    ] {
        let num_elems = shape.iter().product::<usize>() as i64;
        let tensor: TestTensor<4> = TestTensorInt::arange(0..num_elems, &device)
            .reshape(shape)
            .float();
        let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);

        for (i, j) in get_combinations(shape.len()) {
            let view = tensor.clone().swap_dims(i, j);
            let view_ref = tensor_ref.clone().swap_dims(i, j);
            let data = view.into_data();
            let data_ref = view_ref.into_data();

            data_ref.assert_approx_eq::<FloatElem>(&data, Tolerance::default());
        }
    }
}

fn get_combinations(n: usize) -> impl Iterator<Item = (usize, usize)> {
    // Iterate from 0 up to n
    (0..n).flat_map(move |i| {
        // For each i, iterate from i + 1 up to n
        // This ensures no repeats (i == j) and no duplicates (j, i)
        (i + 1..n).map(move |j| (i, j))
    })
}
