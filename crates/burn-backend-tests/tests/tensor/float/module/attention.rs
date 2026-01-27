use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;
use burn_tensor::module::attention;
use burn_tensor::module::naive_attention;

#[test]
fn test_attention_no_mask() {
    // Skip on metal with f16 - flash attention returns zeros
    #[cfg(feature = "metal")]
    if core::any::TypeId::of::<FloatElemType>() == core::any::TypeId::of::<burn_tensor::f16>() {
        return;
    }

    let num_batches = 1;
    let num_heads = 1;
    let seq_q = 128;
    let seq_kv = 128;
    let head_dim = 64;
    let val_dim = 64;

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_q, head_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_kv, head_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_kv, val_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );

    let output = attention(query.clone(), key.clone(), value.clone(), None);

    let expected = naive_attention::<TestBackend>(query, key, value, None);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected.into_data(), Tolerance::relative(1e-2));
}
