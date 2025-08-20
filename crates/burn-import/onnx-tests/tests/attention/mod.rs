// Import the shared macro
use crate::include_models;
include_models!(
    attention_4d,
    attention_3d,
    attention_attn_mask_bool,
    attention_attn_mask_int,
    attention_attn_mask_float,
    attention_softcap,
    attention_cache,
    attention_custom_scale,
    attention_is_causal,
    attention_qk_output_0,
    attention_qk_output_1,
    attention_qk_output_2,
    attention_qk_output_3
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TestBackend;
    use burn::tensor::{Bool, Int, Tensor, TensorData, Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn simple_4d() {
        let device = Default::default();
        let model: attention_4d::Model<TestBackend> = attention_4d::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn simple_3d() {
        let device = Default::default();
        let model: attention_3d::Model<TestBackend> = attention_3d::Model::new(&device);

        let q = Tensor::<TestBackend, 3>::from_floats([[[1.0, 0.0], [0.0, 1.0]]], &device);
        let k = Tensor::<TestBackend, 3>::from_floats([[[0.0, 1.0], [1.0, 0.0]]], &device);
        let v = Tensor::<TestBackend, 3>::from_floats([[[0.25, 0.5], [0.3, 0.6]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[0.283488f32, 0.566976], [0.266511, 0.533023]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_bool() {
        let device = Default::default();
        let model: attention_attn_mask_bool::Model<TestBackend> =
            attention_attn_mask_bool::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, false], [false, true]]),
            &device,
        );

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_int() {
        let device = Default::default();
        let model: attention_attn_mask_int::Model<TestBackend> =
            attention_attn_mask_int::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<TestBackend, 2, Int>::from_ints([[2, 0], [0, 3]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn attn_mask_float() {
        let device = Default::default();
        let model: attention_attn_mask_float::Model<TestBackend> =
            attention_attn_mask_float::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);
        let attn_mask = Tensor::<TestBackend, 2>::from_floats([[2.0, 0.0], [0.0, 3.0]], &device);

        let output = model.forward(q, k, v, attn_mask);
        let expected = TensorData::from([[[[0.260768f32, 0.521536], [0.295414, 0.590828]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn softcap() {
        let device = Default::default();
        let model: attention_softcap::Model<TestBackend> = attention_softcap::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.283176f32, 0.566352], [0.266823, 0.533647]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[allow(clippy::type_complexity)]
    fn cached_attn_inputs() -> (
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 2, Bool>,
        Tensor<TestBackend, 4>,
        Tensor<TestBackend, 4>,
    ) {
        let device = &Default::default();
        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0]]]], device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.3, 0.6]]]], device);
        let attn_mask = Tensor::<TestBackend, 2, Bool>::from_bool(
            TensorData::from([[true, true], [true, true]]),
            device,
        );
        let past_k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0]]]], device);
        let past_v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5]]]], device);

        (q, k, v, attn_mask, past_k, past_v)
    }

    #[test]
    fn cache() {
        let device = Default::default();
        let model: attention_cache::Model<TestBackend> = attention_cache::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (output, present_k, present_v) = model.forward(q, k, v, attn_mask, past_k, past_v);
        let expected = TensorData::from([[[[0.283488f32, 0.566976], [0.266511, 0.533023]]]]);
        let expected_k = TensorData::from([[[[0.0, 1.0], [1.0, 0.0]]]]);
        let expected_v = TensorData::from([[[[0.25, 0.5], [0.3, 0.6]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
        present_k
            .to_data()
            .assert_approx_eq::<FT>(&expected_k, Tolerance::default());
        present_v
            .to_data()
            .assert_approx_eq::<FT>(&expected_v, Tolerance::default());
    }

    #[test]
    fn custom_scale() {
        let device = Default::default();
        let model: attention_custom_scale::Model<TestBackend> =
            attention_custom_scale::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.294039f32, 0.588079], [0.255960, 0.511920]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn is_causal() {
        let device = Default::default();
        let model: attention_is_causal::Model<TestBackend> =
            attention_is_causal::Model::new(&device);

        let q = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 0.0], [0.0, 1.0]]]], &device);
        let k = Tensor::<TestBackend, 4>::from_floats([[[[0.0, 1.0], [1.0, 0.0]]]], &device);
        let v = Tensor::<TestBackend, 4>::from_floats([[[[0.25, 0.5], [0.3, 0.6]]]], &device);

        let output = model.forward(q, k, v);
        let expected = TensorData::from([[[[0.25f32, 0.5], [0.266511, 0.533023]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_0() {
        let device = Default::default();
        let model: attention_qk_output_0::Model<TestBackend> =
            attention_qk_output_0::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.707106], [0.707106, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<FT>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_1() {
        let device = Default::default();
        let model: attention_qk_output_1::Model<TestBackend> =
            attention_qk_output_1::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.707106], [0.707106, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<FT>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_2() {
        let device = Default::default();
        let model: attention_qk_output_2::Model<TestBackend> =
            attention_qk_output_2::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.0f32, 0.67904], [0.67904, 0.0]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<FT>(&expected_qk, Tolerance::default());
    }

    #[test]
    fn qk_matmul_output_3() {
        let device = Default::default();
        let model: attention_qk_output_3::Model<TestBackend> =
            attention_qk_output_3::Model::new(&device);

        let (q, k, v, attn_mask, past_k, past_v) = cached_attn_inputs();

        let (_, _, _, qk_output) = model.forward(q, k, v, attn_mask, past_k, past_v);
        #[allow(clippy::approx_constant)]
        let expected_qk = TensorData::from([[[[0.336474f32, 0.663525], [0.663525, 0.336474]]]]);

        qk_output
            .to_data()
            .assert_approx_eq::<FT>(&expected_qk, Tolerance::default());
    }
}
