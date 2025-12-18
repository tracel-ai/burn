use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Distribution, Int, Shape, Tensor};

#[test]
pub fn into_contiguous_match_reference_backend_1() {
    // for shape in [vec![32, 32, 32, 32], vec![1, 256, 256, 256]] {
    for shape in [[4, 4, 4, 4]] {
        let num_elems = shape.iter().product::<usize>() as i64;
        let tensor: Tensor<TestBackend, 4> =
            Tensor::<TestBackend, 1, Int>::arange(0..num_elems, &Default::default())
                .reshape(shape)
                .float();
        let tensor_ref =
            Tensor::<ReferenceBackend, 4>::from_data(tensor.to_data(), &Default::default());

        // for (i, j) in get_combinations(shape.len()) {
        for (i, j) in [(0, 3)] {
            println!("{i}-{j}");
            let view = tensor.clone().swap_dims(i, j);
            let view_ref = tensor_ref.clone().swap_dims(i, j);
            println!("{view_ref}");
            let data = view.into_data();
            let data_ref = view_ref.into_data();
            println!("{data}");
            println!("{data_ref}");

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
