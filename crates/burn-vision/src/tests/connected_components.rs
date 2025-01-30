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
        let tensor = TestTensorBool::<2>::from(space_invader()).unsqueeze::<4>();

        let output = tensor.connected_components(Connectivity::Eight);
        let expected = space_invader(); // All pixels are in the same group for 8-connected
        let expected = TestTensorInt::<2>::from(expected).unsqueeze::<3>();

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_8_connectivity_with_stats() {
        let tensor = TestTensorBool::<2>::from(space_invader()).unsqueeze::<4>();

        let (output, stats) = tensor
            .connected_components_with_stats(Connectivity::Eight, ConnectedStatsOptions::all());
        let expected = space_invader(); // All pixels are in the same group for 8-connected
        let expected = TestTensorInt::<2>::from(expected).unsqueeze::<3>();

        let (area, left, top, right, bottom) = normalize_stats(
            stats.area.into_data(),
            stats.left.into_data(),
            stats.top.into_data(),
            stats.right.into_data(),
            stats.bottom.into_data(),
        );

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);

        area.assert_eq(&TensorData::from([[58]]), false);
        left.assert_eq(&TensorData::from([[0]]), false);
        top.assert_eq(&TensorData::from([[0]]), false);
        right.assert_eq(&TensorData::from([[13]]), false);
        bottom.assert_eq(&TensorData::from([[8]]), false);
    }

    #[test]
    fn should_support_4_connectivity() {
        let tensor = TestTensorBool::<2>::from(space_invader()).unsqueeze::<4>();

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
        let expected = TestTensorInt::<2>::from(expected).unsqueeze::<3>();

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);
    }

    #[test]
    fn should_support_4_connectivity_with_stats() {
        let tensor = TestTensorBool::<2>::from(space_invader()).unsqueeze::<4>();

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
        let expected = TestTensorInt::<2>::from(expected).unsqueeze::<3>();

        let (area, left, top, right, bottom) = normalize_stats(
            stats.area.into_data(),
            stats.left.into_data(),
            stats.top.into_data(),
            stats.right.into_data(),
            stats.bottom.into_data(),
        );

        normalize_labels(output.into_data()).assert_eq(&expected.into_data(), false);

        area.assert_eq(&TensorData::from([[1, 1, 46, 5, 5]]), false);
        left.assert_eq(&TensorData::from([[3, 10, 1, 0, 12]]), false);
        top.assert_eq(&TensorData::from([[0, 0, 1, 5, 5]]), false);
        right.assert_eq(&TensorData::from([[3, 10, 12, 1, 13]]), false);
        bottom.assert_eq(&TensorData::from([[0, 0, 8, 7, 7]]), false);
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

    fn normalize_stats(
        area: TensorData,
        left: TensorData,
        top: TensorData,
        right: TensorData,
        bottom: TensorData,
    ) -> (TensorData, TensorData, TensorData, TensorData, TensorData) {
        let batches = area.shape[0];

        let area = area.as_slice::<i32>().unwrap();
        let left = left.as_slice::<i32>().unwrap();
        let top = top.as_slice::<i32>().unwrap();
        let right = right.as_slice::<i32>().unwrap();
        let bottom = bottom.as_slice::<i32>().unwrap();

        let mut area_new = vec![];
        let mut left_new = vec![];
        let mut top_new = vec![];
        let mut right_new = vec![];
        let mut bottom_new = vec![];

        for (label, area) in area.iter().enumerate() {
            if *area != 0 {
                area_new.push(*area);
                left_new.push(left[label]);
                top_new.push(top[label]);
                right_new.push(right[label]);
                bottom_new.push(bottom[label]);
            }
        }

        let shape = [batches, area_new.len() / batches];

        (
            TensorData::new(area_new, shape.clone()),
            TensorData::new(left_new, shape.clone()),
            TensorData::new(top_new, shape),
            TensorData::new(right_new, shape.clone()),
            TensorData::new(bottom_new, shape.clone()),
        )
    }
}
