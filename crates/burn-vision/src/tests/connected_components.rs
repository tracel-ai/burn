#[burn_tensor_testgen::testgen(connected_components)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use burn_tensor::TensorData;
    use burn_vision::{as_type, ConnectedComponents, ConnectedStatsOptions, Connectivity};

    fn space_invader() -> [[IntType; 14]; 9] {
        as_type!(IntType: [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ])
    }

    #[test]
    fn should_support_8_connectivity() {
        let tensor = TestTensorBool::<2>::from(space_invader());

        let output = tensor.connected_components(Connectivity::Eight);
        let expected = space_invader(); // All pixels are in the same group for 8-connected
        let expected = TestTensorInt::<2>::from(expected);

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_8_connectivity_with_stats() {
        let tensor = TestTensorBool::<2>::from(space_invader());

        let (output, stats) = tensor
            .connected_components_with_stats(Connectivity::Eight, ConnectedStatsOptions::all());
        let expected = space_invader(); // All pixels are in the same group for 8-connected
        let expected = TestTensorInt::<2>::from(expected);

        let (area, left, top, right, bottom) = (
            stats.area.slice([1..2]).into_data(),
            stats.left.slice([1..2]).into_data(),
            stats.top.slice([1..2]).into_data(),
            stats.right.slice([1..2]).into_data(),
            stats.bottom.slice([1..2]).into_data(),
        );

        output.into_data().assert_eq(&expected.into_data(), false);

        area.assert_eq(&TensorData::from([58]), false);
        left.assert_eq(&TensorData::from([0]), false);
        top.assert_eq(&TensorData::from([0]), false);
        right.assert_eq(&TensorData::from([13]), false);
        bottom.assert_eq(&TensorData::from([8]), false);
        stats
            .max_label
            .into_data()
            .assert_eq(&TensorData::from([1]), false);
    }

    #[test]
    fn should_support_4_connectivity() {
        let tensor = TestTensorBool::<2>::from(space_invader());

        let output = tensor.connected_components(Connectivity::Four);
        let expected = as_type!(IntType: [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [4, 0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0, 5],
            [4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 5, 5],
            [4, 4, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 5, 5],
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        ]);
        let expected = TestTensorInt::<2>::from(expected);

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_4_connectivity_with_stats() {
        let tensor = TestTensorBool::<2>::from(space_invader());

        let (output, stats) = tensor
            .connected_components_with_stats(Connectivity::Four, ConnectedStatsOptions::all());
        let expected = as_type!(IntType: [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 3, 3, 0, 0, 3, 3, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [4, 0, 0, 3, 3, 0, 0, 0, 0, 3, 3, 0, 0, 5],
            [4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 5, 5],
            [4, 4, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 5, 5],
            [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
        ]);
        let expected = TestTensorInt::<2>::from(expected);

        // Slice off background and limit to compacted labels
        let (area, left, top, right, bottom) = (
            stats.area.slice([1..6]).into_data(),
            stats.left.slice([1..6]).into_data(),
            stats.top.slice([1..6]).into_data(),
            stats.right.slice([1..6]).into_data(),
            stats.bottom.slice([1..6]).into_data(),
        );

        output.into_data().assert_eq(&expected.into_data(), false);

        area.assert_eq(&TensorData::from([1, 1, 46, 5, 5]), false);
        left.assert_eq(&TensorData::from([3, 10, 1, 0, 12]), false);
        top.assert_eq(&TensorData::from([0, 0, 1, 5, 5]), false);
        right.assert_eq(&TensorData::from([3, 10, 12, 1, 13]), false);
        bottom.assert_eq(&TensorData::from([0, 0, 8, 7, 7]), false);
        stats
            .max_label
            .into_data()
            .assert_eq(&TensorData::from([5]), false);
    }

    /// Normalize labels to sequential since actual labels aren't required to be contiguous and
    /// different algorithms can return different numbers even if correct
    fn normalize_labels(mut labels: TensorData) -> TensorData {
        let mut next_label = 0;
        let mut mappings = HashMap::<i32, i32>::default();
        let data = labels.as_mut_slice::<i32>().unwrap();
        for label in data {
            if *label != 0 {
                let relabel = mappings.entry(*label).or_insert_with(|| {
                    next_label += 1;
                    next_label
                });
                *label = *relabel;
            }
        }
        labels
    }
}
