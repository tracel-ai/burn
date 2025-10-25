extern crate alloc;

// Disabled for now because https://github.com/huggingface/safetensors/issues/650 is not published
// #[test]
// fn test_safetensors_no_std() {
//     use burn_ndarray::NdArray;
//     use burn_no_std_tests::safetensors;
//     type Backend = NdArray<f32>;
//     let device = Default::default();

//     // Run all SafeTensors tests
//     safetensors::run_all_tests::<Backend>(&device);
// }
