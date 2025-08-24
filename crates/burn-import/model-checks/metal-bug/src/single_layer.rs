use burn::prelude::*;
use burn::nn::*;
use burn::nn::conv::*;
use burn::nn::pool::*;
use burn::module::Param;

pub type NdarrayBackend = burn::backend::NdArray;
pub type MetalBackend = burn::backend::Metal;
pub type TchBackend = burn::backend::LibTorch<f32>;

fn test_linear<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    // Test Linear layer with uniform weights
    let config = LinearConfig::new(256, 512);
    let mut linear1 = config.init::<B1>(device1);
    let mut linear2 = config.init::<B2>(device2);
    
    // Set same weights
    let weight_data = vec![0.01f32; 256 * 512];
    let bias_data = vec![0.01f32; 512];
    
    // In Burn, Linear weight shape is [d_input, d_output]
    let weight1 = Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1).reshape([256, 512]);
    let weight2 = Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2).reshape([256, 512]);
    linear1.weight = Param::from_tensor(weight1);
    linear2.weight = Param::from_tensor(weight2);
    
    if linear1.bias.is_some() {
        let bias1 = Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1);
        linear1.bias = Some(Param::from_tensor(bias1));
    }
    if linear2.bias.is_some() {
        let bias2 = Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2);
        linear2.bias = Some(Param::from_tensor(bias2));
    }
    
    // Test with ones input
    let input1 = Tensor::<B1, 2>::ones([4, 256], device1);
    let input2 = Tensor::<B2, 2>::ones([4, 256], device2);
    
    let output1 = linear1.forward(input1);
    let output2 = linear2.forward(input2);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Linear {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_conv1d<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    let config = Conv1dConfig::new(32, 64, 3);
    let mut conv1 = config.init::<B1>(device1);
    let mut conv2 = config.init::<B2>(device2);
    
    // Set uniform weights AND bias
    let weight_data = vec![0.01f32; 64 * 32 * 3];
    let bias_data = vec![0.01f32; 64];
    let weight1 = Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1).reshape([64, 32, 3]);
    let weight2 = Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2).reshape([64, 32, 3]);
    conv1.weight = Param::from_tensor(weight1);
    conv2.weight = Param::from_tensor(weight2);
    
    if conv1.bias.is_some() {
        let bias1 = Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1);
        conv1.bias = Some(Param::from_tensor(bias1));
    }
    if conv2.bias.is_some() {
        let bias2 = Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2);
        conv2.bias = Some(Param::from_tensor(bias2));
    }
    
    let input1 = Tensor::<B1, 3>::ones([1, 32, 128], device1);
    let input2 = Tensor::<B2, 3>::ones([1, 32, 128], device2);
    
    let output1 = conv1.forward(input1);
    let output2 = conv2.forward(input2);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Conv1d {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_convtranspose2d<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    let config = ConvTranspose2dConfig::new([32, 16], [4, 4])
        .with_stride([2, 2]);
    let mut conv1 = config.init::<B1>(device1);
    let mut conv2 = config.init::<B2>(device2);
    
    // Set uniform weights AND bias
    let weight_data = vec![0.01f32; 32 * 16 * 4 * 4];
    let bias_data = vec![0.01f32; 16];
    let weight1 = Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1).reshape([32, 16, 4, 4]);
    let weight2 = Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2).reshape([32, 16, 4, 4]);
    conv1.weight = Param::from_tensor(weight1);
    conv2.weight = Param::from_tensor(weight2);
    
    if conv1.bias.is_some() {
        let bias1 = Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1);
        conv1.bias = Some(Param::from_tensor(bias1));
    }
    if conv2.bias.is_some() {
        let bias2 = Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2);
        conv2.bias = Some(Param::from_tensor(bias2));
    }
    
    let input1 = Tensor::<B1, 4>::ones([1, 32, 32, 32], device1);
    let input2 = Tensor::<B2, 4>::ones([1, 32, 32, 32], device2);
    
    let output1 = conv1.forward(input1);
    let output2 = conv2.forward(input2);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("ConvTranspose2d {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_maxpool2d<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    // Test MaxPool2d with kernel size 2x2
    let pool1 = MaxPool2dConfig::new([2, 2])
        .with_strides([2, 2])
        .init();
    let pool2 = MaxPool2dConfig::new([2, 2])
        .with_strides([2, 2])
        .init();
    
    // Create test input with pattern
    let mut input_data = vec![0.0f32; 32 * 64 * 64];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 200) as f32 - 100.0) * 0.01;
    }
    
    let input1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 32, 64, 64]);
    let input2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 32, 64, 64]);
    
    let output1 = pool1.forward(input1);
    let output2 = pool2.forward(input2);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("MaxPool2d {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_avgpool2d<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    // Test AvgPool2d with kernel size 2x2
    let pool1 = AvgPool2dConfig::new([2, 2])
        .with_strides([2, 2])
        .init();
    let pool2 = AvgPool2dConfig::new([2, 2])
        .with_strides([2, 2])
        .init();
    
    // Create test input with pattern
    let mut input_data = vec![0.0f32; 32 * 64 * 64];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 200) as f32 - 100.0) * 0.01;
    }
    
    let input1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 32, 64, 64]);
    let input2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 32, 64, 64]);
    
    let output1 = pool1.forward(input1);
    let output2 = pool2.forward(input2);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("AvgPool2d {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_interpolate<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    // Test bilinear interpolation (upsample 2x)
    use burn::tensor::ops::InterpolateMode;
    use burn::tensor::ops::InterpolateOptions;
    
    // Create test input
    let mut input_data = vec![0.0f32; 32 * 32 * 32];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 200) as f32 - 100.0) * 0.01;
    }
    
    let input1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 32, 32, 32]);
    let input2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 32, 32, 32]);
    
    // Upsample to 64x64 using bilinear interpolation
    let options = InterpolateOptions {
        mode: InterpolateMode::Bilinear,
    };
    
    let output1 = burn::tensor::module::interpolate(input1, [64, 64], options.clone());
    let output2 = burn::tensor::module::interpolate(input2, [64, 64], options);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Interpolate(Bilinear) {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_interpolate_nearest<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) -> f32 {
    // Test nearest neighbor interpolation (upsample 2x)
    use burn::tensor::ops::InterpolateMode;
    use burn::tensor::ops::InterpolateOptions;
    
    // Create test input
    let mut input_data = vec![0.0f32; 32 * 32 * 32];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 200) as f32 - 100.0) * 0.01;
    }
    
    let input1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 32, 32, 32]);
    let input2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 32, 32, 32]);
    
    // Upsample to 64x64 using nearest neighbor
    let options = InterpolateOptions {
        mode: InterpolateMode::Nearest,
    };
    
    let output1 = burn::tensor::module::interpolate(input1, [64, 64], options.clone());
    let output2 = burn::tensor::module::interpolate(input2, [64, 64], options);
    
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let max_diff = data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Interpolate(Nearest) {} vs {}: max_diff = {:.8}", name1, name2, max_diff);
    max_diff
}

fn test_activations<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
) {
    println!("\nTesting activations {} vs {}:", name1, name2);
    
    // Create test input with various values
    let mut input_data = vec![0.0f32; 1000];
    for i in 0..1000 {
        input_data[i] = (i as f32 - 500.0) * 0.01; // Range -5 to 5
    }
    
    let input1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1);
    let input2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2);
    
    // Test ReLU
    let relu1 = burn::tensor::activation::relu(input1.clone());
    let relu2 = burn::tensor::activation::relu(input2.clone());
    let relu_diff = compare_outputs(relu1, relu2);
    println!("  ReLU: {:.8}", relu_diff);
    
    // Test Sigmoid
    let sig1 = burn::tensor::activation::sigmoid(input1.clone());
    let sig2 = burn::tensor::activation::sigmoid(input2.clone());
    let sig_diff = compare_outputs(sig1, sig2);
    println!("  Sigmoid: {:.8}", sig_diff);
    
    // Test Tanh
    let tanh1 = burn::tensor::activation::tanh(input1.clone());
    let tanh2 = burn::tensor::activation::tanh(input2.clone());
    let tanh_diff = compare_outputs(tanh1, tanh2);
    println!("  Tanh: {:.8}", tanh_diff);
    
    // Test GELU
    let gelu1 = burn::tensor::activation::gelu(input1.clone());
    let gelu2 = burn::tensor::activation::gelu(input2.clone());
    let gelu_diff = compare_outputs(gelu1, gelu2);
    println!("  GELU: {:.8}", gelu_diff);
    
    // Test SiLU (x * sigmoid(x))
    let silu1 = input1.clone().mul(burn::tensor::activation::sigmoid(input1));
    let silu2 = input2.clone().mul(burn::tensor::activation::sigmoid(input2));
    let silu_diff = compare_outputs(silu1, silu2);
    println!("  SiLU: {:.8}", silu_diff);
}

fn compare_outputs<B1: Backend, B2: Backend>(
    output1: Tensor<B1, 1>,
    output2: Tensor<B2, 1>,
) -> f32 {
    let data1 = output1.into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = output2.into_data().as_slice::<f32>().unwrap().to_vec();
    
    data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("========================================");
    println!("Critical Layer Backend Testing");
    println!("========================================\n");
    
    let ndarray_device = <NdarrayBackend as Backend>::Device::default();
    let metal_device = <MetalBackend as Backend>::Device::default();
    let tch_device = <TchBackend as Backend>::Device::default();
    
    println!("Testing layers with uniform weights (triggers Conv2d bug):\n");
    
    // Test Linear
    println!("\n📊 Linear Layer:");
    println!("----------------------------------------");
    let linear_metal = test_linear::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let linear_ndarray = test_linear::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test Conv1d
    println!("\n📊 Conv1d Layer:");
    println!("----------------------------------------");
    let conv1d_metal = test_conv1d::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let conv1d_ndarray = test_conv1d::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test Conv2d (we know this fails)
    println!("\n📊 Conv2d Layer (YOLO config):");
    println!("----------------------------------------");
    let config = Conv2dConfig::new([3, 96], [3, 3])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1));
    
    let mut conv2d_tch = config.init::<TchBackend>(&tch_device);
    let mut conv2d_metal = config.init::<MetalBackend>(&metal_device);
    let mut conv2d_ndarray = config.init::<NdarrayBackend>(&ndarray_device);
    
    // Set uniform weights (triggers bug)
    let weight_data = vec![0.01f32; 96 * 3 * 3 * 3];
    let weight_tch = Tensor::<TchBackend, 1>::from_data(weight_data.as_slice(), &tch_device)
        .reshape([96, 3, 3, 3]);
    let weight_metal = Tensor::<MetalBackend, 1>::from_data(weight_data.as_slice(), &metal_device)
        .reshape([96, 3, 3, 3]);
    let weight_ndarray = Tensor::<NdarrayBackend, 1>::from_data(weight_data.as_slice(), &ndarray_device)
        .reshape([96, 3, 3, 3]);
    
    conv2d_tch.weight = Param::from_tensor(weight_tch);
    conv2d_metal.weight = Param::from_tensor(weight_metal);
    conv2d_ndarray.weight = Param::from_tensor(weight_ndarray);
    
    // Set bias explicitly
    let bias_data = vec![0.01f32; 96];
    let bias_tch = Tensor::<TchBackend, 1>::from_data(bias_data.as_slice(), &tch_device);
    let bias_metal = Tensor::<MetalBackend, 1>::from_data(bias_data.as_slice(), &metal_device);
    let bias_ndarray = Tensor::<NdarrayBackend, 1>::from_data(bias_data.as_slice(), &ndarray_device);
    
    conv2d_tch.bias = Some(Param::from_tensor(bias_tch));
    conv2d_metal.bias = Some(Param::from_tensor(bias_metal));
    conv2d_ndarray.bias = Some(Param::from_tensor(bias_ndarray));
    
    let input_tch = Tensor::<TchBackend, 4>::ones([1, 3, 72, 72], &tch_device);
    let input_metal = Tensor::<MetalBackend, 4>::ones([1, 3, 72, 72], &metal_device);
    let input_ndarray = Tensor::<NdarrayBackend, 4>::ones([1, 3, 72, 72], &ndarray_device);
    
    let output_tch = conv2d_tch.forward(input_tch);
    let output_metal = conv2d_metal.forward(input_metal);
    let output_ndarray = conv2d_ndarray.forward(input_ndarray);
    
    let tch_data = output_tch.into_data().as_slice::<f32>().unwrap().to_vec();
    let metal_data = output_metal.into_data().as_slice::<f32>().unwrap().to_vec();
    let ndarray_data = output_ndarray.into_data().as_slice::<f32>().unwrap().to_vec();
    
    let conv2d_metal_diff = tch_data.iter().zip(metal_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let conv2d_ndarray_diff = tch_data.iter().zip(ndarray_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    println!("Conv2d tch vs metal: max_diff = {:.8}", conv2d_metal_diff);
    println!("Conv2d tch vs ndarray: max_diff = {:.8}", conv2d_ndarray_diff);
    
    // Test ConvTranspose2d
    println!("\n📊 ConvTranspose2d Layer:");
    println!("----------------------------------------");
    let convt_metal = test_convtranspose2d::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let convt_ndarray = test_convtranspose2d::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test MaxPool2d
    println!("\n📊 MaxPool2d Layer:");
    println!("----------------------------------------");
    let maxpool_metal = test_maxpool2d::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let maxpool_ndarray = test_maxpool2d::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test AvgPool2d
    println!("\n📊 AvgPool2d Layer:");
    println!("----------------------------------------");
    let avgpool_metal = test_avgpool2d::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let avgpool_ndarray = test_avgpool2d::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test Interpolate (Bilinear)
    println!("\n📊 Interpolate Layer:");
    println!("----------------------------------------");
    let interp_metal = test_interpolate::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let interp_ndarray = test_interpolate::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test Interpolate (Nearest)
    let interp_nearest_metal = test_interpolate_nearest::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    let interp_nearest_ndarray = test_interpolate_nearest::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Test Activations
    println!("\n📊 Activation Functions:");
    println!("----------------------------------------");
    test_activations::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device
    );
    test_activations::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device
    );
    
    // Summary
    println!("\n========================================");
    println!("SUMMARY");
    println!("========================================\n");
    
    let threshold = 0.0001;
    
    println!("Layer               | Metal vs Tch    | Ndarray vs Tch  | Status");
    println!("--------------------|-----------------|-----------------|--------");
    println!("Linear              | {:.8} | {:.8} | {}", 
             linear_metal, linear_ndarray,
             if linear_metal > threshold || linear_ndarray > threshold { "❌" } else { "✅" });
    println!("Conv1d              | {:.8} | {:.8} | {}", 
             conv1d_metal, conv1d_ndarray,
             if conv1d_metal > threshold || conv1d_ndarray > threshold { "❌" } else { "✅" });
    println!("Conv2d              | {:.8} | {:.8} | {}", 
             conv2d_metal_diff, conv2d_ndarray_diff,
             if conv2d_metal_diff > threshold || conv2d_ndarray_diff > threshold { "❌" } else { "✅" });
    println!("ConvTranspose2d     | {:.8} | {:.8} | {}", 
             convt_metal, convt_ndarray,
             if convt_metal > threshold || convt_ndarray > threshold { "❌" } else { "✅" });
    println!("MaxPool2d           | {:.8} | {:.8} | {}", 
             maxpool_metal, maxpool_ndarray,
             if maxpool_metal > threshold || maxpool_ndarray > threshold { "❌" } else { "✅" });
    println!("AvgPool2d           | {:.8} | {:.8} | {}", 
             avgpool_metal, avgpool_ndarray,
             if avgpool_metal > threshold || avgpool_ndarray > threshold { "❌" } else { "✅" });
    println!("Interpolate(Bilin)  | {:.8} | {:.8} | {}", 
             interp_metal, interp_ndarray,
             if interp_metal > threshold || interp_ndarray > threshold { "❌" } else { "✅" });
    println!("Interpolate(Near)   | {:.8} | {:.8} | {}", 
             interp_nearest_metal, interp_nearest_ndarray,
             if interp_nearest_metal > threshold || interp_nearest_ndarray > threshold { "❌" } else { "✅" });
    
    println!("\n⚠️ Threshold for failure: {}", threshold);
    
    if conv2d_metal_diff > threshold {
        println!("\n🔴 CRITICAL: Conv2d shows massive divergence on Metal backend!");
    }
    if conv2d_ndarray_diff > threshold {
        println!("🔴 CRITICAL: Conv2d shows divergence on Ndarray backend!");
    }
}