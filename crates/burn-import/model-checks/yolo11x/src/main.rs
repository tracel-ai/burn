extern crate alloc;

// Import the generated model code as a module
#[allow(clippy::type_complexity)]
pub mod yolo11x_opset16 {
    include!(concat!(env!("OUT_DIR"), "/model/yolo11x_opset16.rs"));
}

#[cfg(all(feature = "ndarray", not(feature = "tch")))]
type MyBackend = burn::backend::NdArray<f32>;

#[cfg(feature = "tch")]
type MyBackend = burn::backend::LibTorch;

fn main() {
    println!("========================================");
    println!("YOLO11x Burn Model Test");
    println!("========================================\n");

    // Initialize the model (without weights for now)
    println!("Initializing YOLO11x model...");
    let device = Default::default();
    let model: yolo11x_opset16::Model<MyBackend> = yolo11x_opset16::Model::default();

    // Create a test input tensor with the expected shape [1, 3, 640, 640]
    println!("Creating test input tensor [1, 3, 640, 640]...");
    let input = burn::tensor::Tensor::<MyBackend, 4>::random(
        [1, 3, 640, 640],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // Run inference
    println!("Running model inference...");
    let output = model.forward(input);

    // Display output shape
    let shape = output.shape();
    println!("\n✓ Inference successful!");
    println!("  Output shape: {:?}", shape.dims);

    // Verify expected output shape
    let expected_shape = [1, 84, 8400];
    if shape.dims == expected_shape {
        println!("  ✓ Output shape matches expected: {:?}", expected_shape);
    } else {
        println!(
            "  ⚠ Warning: Expected shape {:?}, got {:?}",
            expected_shape, shape.dims
        );
    }

    println!("\n========================================");
    println!("Model test completed successfully!");
    println!("========================================");
}
