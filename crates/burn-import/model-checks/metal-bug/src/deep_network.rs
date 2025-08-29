use burn::prelude::*;
use burn::nn::*;
use burn::nn::conv::*;
use burn::nn::pool::*;
use burn::module::Param;

pub type NdarrayBackend = burn::backend::NdArray;
pub type MetalBackend = burn::backend::Metal;
pub type TchBackend = burn::backend::LibTorch<f32>;

/// Build a deep CNN-like network and test error accumulation
fn test_deep_cnn<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
    depth: usize,
) -> Vec<f32> {
    println!("\n🏗️ Building {}-layer deep CNN for {} vs {}", depth, name1, name2);
    println!("  Input: [1, 3, 128, 128]");
    
    // Track differences at each layer
    let mut layer_diffs = Vec::new();
    
    // Start with same input
    let mut input_data = vec![0.0f32; 3 * 128 * 128];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 200) as f32 - 100.0) * 0.01; // Range -1 to 1
    }
    
    let mut tensor1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 3, 128, 128]);
    let mut tensor2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 3, 128, 128]);
    
    let mut current_channels = 3;
    let mut current_size = 128;
    
    for layer_idx in 0..depth {
        // Alternate between different layer types for realistic network
        let layer_type = layer_idx % 4;
        
        match layer_type {
            0 | 2 => {
                // Conv2d layer (increase channels, decrease spatial size)
                let out_channels = if current_channels < 512 { 
                    current_channels * 2 
                } else { 
                    current_channels 
                };
                
                let config = Conv2dConfig::new([current_channels, out_channels], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Same);
                
                let mut conv1 = config.init::<B1>(device1);
                let mut conv2 = config.init::<B2>(device2);
                
                // Set deterministic weights and bias
                let weight_data = vec![0.01f32; out_channels * current_channels * 3 * 3];
                let bias_data = vec![0.01f32; out_channels];
                
                conv1.weight = Param::from_tensor(
                    Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1)
                        .reshape([out_channels, current_channels, 3, 3])
                );
                conv2.weight = Param::from_tensor(
                    Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2)
                        .reshape([out_channels, current_channels, 3, 3])
                );
                
                conv1.bias = Some(Param::from_tensor(
                    Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1)
                ));
                conv2.bias = Some(Param::from_tensor(
                    Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2)
                ));
                
                tensor1 = conv1.forward(tensor1);
                tensor2 = conv2.forward(tensor2);
                
                // Apply ReLU activation
                tensor1 = burn::tensor::activation::relu(tensor1);
                tensor2 = burn::tensor::activation::relu(tensor2);
                
                current_channels = out_channels;
                
                // Calculate difference
                let diff = calculate_difference(&tensor1, &tensor2);
                layer_diffs.push(diff);
                
                println!("  Layer {:2} (Conv2d {}->{}): max_diff = {:.8}", 
                         layer_idx + 1, current_channels/2, current_channels, diff);
            }
            1 => {
                // MaxPool2d (reduce spatial size)
                if current_size > 8 {
                    let pool1 = MaxPool2dConfig::new([2, 2])
                        .with_strides([2, 2])
                        .init();
                    let pool2 = MaxPool2dConfig::new([2, 2])
                        .with_strides([2, 2])
                        .init();
                    
                    tensor1 = pool1.forward(tensor1);
                    tensor2 = pool2.forward(tensor2);
                    
                    current_size /= 2;
                    
                    let diff = calculate_difference(&tensor1, &tensor2);
                    layer_diffs.push(diff);
                    
                    println!("  Layer {:2} (MaxPool2d): max_diff = {:.8}, size now {}x{}", 
                             layer_idx + 1, diff, current_size, current_size);
                } else {
                    // Skip pooling if too small
                    continue;
                }
            }
            3 => {
                // Instance Normalization instead of BatchNorm (works with any batch size)
                let config = InstanceNormConfig::new(current_channels);
                let in1 = config.init::<B1>(device1);
                let in2 = config.init::<B2>(device2);
                
                tensor1 = in1.forward(tensor1);
                tensor2 = in2.forward(tensor2);
                
                let diff = calculate_difference(&tensor1, &tensor2);
                layer_diffs.push(diff);
                
                println!("  Layer {:2} (InstanceNorm): max_diff = {:.8}", layer_idx + 1, diff);
            }
            _ => unreachable!()
        }
    }
    
    layer_diffs
}

/// Test deep residual-like network with skip connections
fn test_deep_residual<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
    num_blocks: usize,
) -> Vec<f32> {
    println!("\n🏗️ Building {}-block ResNet-like network for {} vs {}", num_blocks, name1, name2);
    
    let mut block_diffs = Vec::new();
    
    // Initial input
    let mut input_data = vec![0.0f32; 64 * 32 * 32];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 100) as f32 - 50.0) * 0.01;
    }
    
    let mut tensor1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 64, 32, 32]);
    let mut tensor2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 64, 32, 32]);
    
    for block_idx in 0..num_blocks {
        // Save input for residual connection
        let residual1 = tensor1.clone();
        let residual2 = tensor2.clone();
        
        // First conv in block
        let config = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same);
        
        let mut conv1_a = config.init::<B1>(device1);
        let mut conv2_a = config.init::<B2>(device2);
        
        set_conv_weights(&mut conv1_a, &mut conv2_a, 64, 64, device1, device2);
        
        tensor1 = conv1_a.forward(tensor1);
        tensor2 = conv2_a.forward(tensor2);
        
        // ReLU
        tensor1 = burn::tensor::activation::relu(tensor1);
        tensor2 = burn::tensor::activation::relu(tensor2);
        
        // Second conv in block
        let mut conv1_b = config.init::<B1>(device1);
        let mut conv2_b = config.init::<B2>(device2);
        
        set_conv_weights(&mut conv1_b, &mut conv2_b, 64, 64, device1, device2);
        
        tensor1 = conv1_b.forward(tensor1);
        tensor2 = conv2_b.forward(tensor2);
        
        // Add residual connection
        tensor1 = tensor1 + residual1;
        tensor2 = tensor2 + residual2;
        
        // Final ReLU
        tensor1 = burn::tensor::activation::relu(tensor1);
        tensor2 = burn::tensor::activation::relu(tensor2);
        
        let diff = calculate_difference(&tensor1, &tensor2);
        block_diffs.push(diff);
        
        println!("  Block {:2}: max_diff = {:.8}", block_idx + 1, diff);
    }
    
    block_diffs
}

/// Test deep fully connected network
fn test_deep_mlp<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
    depth: usize,
) -> Vec<f32> {
    println!("\n🏗️ Building {}-layer deep MLP for {} vs {}", depth, name1, name2);
    
    let mut layer_diffs = Vec::new();
    
    // Start with input
    let mut input_data = vec![0.0f32; 512];
    for i in 0..512 {
        input_data[i] = ((i % 100) as f32 - 50.0) * 0.01;
    }
    
    let mut tensor1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, 512]);
    let mut tensor2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, 512]);
    
    for layer_idx in 0..depth {
        let config = LinearConfig::new(512, 512);
        let mut linear1 = config.init::<B1>(device1);
        let mut linear2 = config.init::<B2>(device2);
        
        // Set same weights and bias
        let weight_data = vec![0.002f32; 512 * 512]; // Smaller weights for stability
        let bias_data = vec![0.001f32; 512];
        
        linear1.weight = Param::from_tensor(
            Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1).reshape([512, 512])
        );
        linear2.weight = Param::from_tensor(
            Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2).reshape([512, 512])
        );
        
        linear1.bias = Some(Param::from_tensor(
            Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1)
        ));
        linear2.bias = Some(Param::from_tensor(
            Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2)
        ));
        
        tensor1 = linear1.forward(tensor1);
        tensor2 = linear2.forward(tensor2);
        
        // Apply activation (alternate between ReLU and GELU)
        if layer_idx % 2 == 0 {
            tensor1 = burn::tensor::activation::relu(tensor1);
            tensor2 = burn::tensor::activation::relu(tensor2);
        } else {
            tensor1 = burn::tensor::activation::gelu(tensor1);
            tensor2 = burn::tensor::activation::gelu(tensor2);
        }
        
        let diff = calculate_difference_2d(&tensor1, &tensor2);
        layer_diffs.push(diff);
        
        println!("  Layer {:2} (Linear + {}): max_diff = {:.8}", 
                 layer_idx + 1, 
                 if layer_idx % 2 == 0 { "ReLU" } else { "GELU" },
                 diff);
    }
    
    layer_diffs
}

/// Test attention mechanism stacking
fn test_deep_attention<B1: Backend, B2: Backend>(
    name1: &str,
    device1: &B1::Device,
    name2: &str,
    device2: &B2::Device,
    num_layers: usize,
) -> Vec<f32> {
    println!("\n🏗️ Building {}-layer attention stack for {} vs {}", num_layers, name1, name2);
    
    let mut layer_diffs = Vec::new();
    let seq_len = 16;
    let embed_dim = 128;
    
    // Create input sequence
    let mut input_data = vec![0.0f32; seq_len * embed_dim];
    for i in 0..input_data.len() {
        input_data[i] = ((i % 100) as f32 - 50.0) * 0.01;
    }
    
    let mut tensor1 = Tensor::<B1, 1>::from_data(input_data.as_slice(), device1)
        .reshape([1, seq_len, embed_dim]);
    let mut tensor2 = Tensor::<B2, 1>::from_data(input_data.as_slice(), device2)
        .reshape([1, seq_len, embed_dim]);
    
    for layer_idx in 0..num_layers {
        // Simple self-attention
        let scale = (embed_dim as f32).sqrt();
        
        // Q, K, V are same (self-attention)
        let q1 = tensor1.clone();
        let k1 = tensor1.clone();
        let v1 = tensor1.clone();
        
        let q2 = tensor2.clone();
        let k2 = tensor2.clone();
        let v2 = tensor2.clone();
        
        // Compute attention scores
        let scores1 = q1.matmul(k1.transpose()) / scale;
        let scores2 = q2.matmul(k2.transpose()) / scale;
        
        // Softmax
        let attn1 = burn::tensor::activation::softmax(scores1, 2);
        let attn2 = burn::tensor::activation::softmax(scores2, 2);
        
        // Apply attention to values
        let output1 = attn1.matmul(v1);
        let output2 = attn2.matmul(v2);
        
        // Add residual connection
        tensor1 = tensor1 + output1;
        tensor2 = tensor2 + output2;
        
        // Layer norm
        let ln_config = LayerNormConfig::new(embed_dim);
        let ln1 = ln_config.init::<B1>(device1);
        let ln2 = ln_config.init::<B2>(device2);
        
        tensor1 = ln1.forward(tensor1);
        tensor2 = ln2.forward(tensor2);
        
        let diff = calculate_difference_3d(&tensor1, &tensor2);
        layer_diffs.push(diff);
        
        println!("  Layer {:2} (Self-Attention + LayerNorm): max_diff = {:.8}", 
                 layer_idx + 1, diff);
    }
    
    layer_diffs
}

// Helper functions
fn calculate_difference<B1: Backend, B2: Backend>(
    tensor1: &Tensor<B1, 4>,
    tensor2: &Tensor<B2, 4>,
) -> f32 {
    let data1 = tensor1.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = tensor2.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    
    data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn calculate_difference_2d<B1: Backend, B2: Backend>(
    tensor1: &Tensor<B1, 2>,
    tensor2: &Tensor<B2, 2>,
) -> f32 {
    let data1 = tensor1.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = tensor2.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    
    data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn calculate_difference_3d<B1: Backend, B2: Backend>(
    tensor1: &Tensor<B1, 3>,
    tensor2: &Tensor<B2, 3>,
) -> f32 {
    let data1 = tensor1.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    let data2 = tensor2.clone().into_data().as_slice::<f32>().unwrap().to_vec();
    
    data1.iter().zip(data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn set_conv_weights<B1: Backend, B2: Backend>(
    conv1: &mut Conv2d<B1>,
    conv2: &mut Conv2d<B2>,
    in_ch: usize,
    out_ch: usize,
    device1: &B1::Device,
    device2: &B2::Device,
) {
    let weight_data = vec![0.01f32; out_ch * in_ch * 3 * 3];
    let bias_data = vec![0.01f32; out_ch];
    
    conv1.weight = Param::from_tensor(
        Tensor::<B1, 1>::from_data(weight_data.as_slice(), device1)
            .reshape([out_ch, in_ch, 3, 3])
    );
    conv2.weight = Param::from_tensor(
        Tensor::<B2, 1>::from_data(weight_data.as_slice(), device2)
            .reshape([out_ch, in_ch, 3, 3])
    );
    
    conv1.bias = Some(Param::from_tensor(
        Tensor::<B1, 1>::from_data(bias_data.as_slice(), device1)
    ));
    conv2.bias = Some(Param::from_tensor(
        Tensor::<B2, 1>::from_data(bias_data.as_slice(), device2)
    ));
}

fn analyze_error_growth(diffs: &[f32], network_type: &str) {
    if diffs.len() < 2 {
        return;
    }
    
    println!("\n📈 Error Growth Analysis for {}:", network_type);
    
    // Calculate growth rate
    let mut max_growth = 0.0f32;
    let mut max_growth_layer = 0;
    
    for i in 1..diffs.len() {
        if diffs[i-1] > 0.0 {
            let growth = diffs[i] / diffs[i-1];
            if growth > max_growth {
                max_growth = growth;
                max_growth_layer = i;
            }
        }
    }
    
    println!("  Initial error: {:.8}", diffs[0]);
    println!("  Final error:   {:.8}", diffs[diffs.len()-1]);
    println!("  Total growth:  {:.2}x", 
             if diffs[0] > 0.0 { diffs[diffs.len()-1] / diffs[0] } else { 0.0 });
    println!("  Max growth:    {:.2}x at layer {}", max_growth, max_growth_layer + 1);
    
    // Check if error is growing exponentially
    let avg_growth = if diffs[0] > 0.0 {
        (diffs[diffs.len()-1] / diffs[0]).powf(1.0 / diffs.len() as f32)
    } else {
        0.0
    };
    
    if avg_growth > 1.1 {
        println!("  ⚠️ WARNING: Exponential error growth detected! (avg: {:.2}x per layer)", avg_growth);
    } else if avg_growth > 1.01 {
        println!("  ⚠️ Linear error accumulation (avg: {:.4}x per layer)", avg_growth);
    } else {
        println!("  ✅ Error is stable or decreasing");
    }
}

fn main() {
    println!("========================================");
    println!("Deep Network Error Accumulation Test");
    println!("========================================\n");
    println!("Testing how errors accumulate through deep networks\n");
    
    let ndarray_device = <NdarrayBackend as Backend>::Device::default();
    let metal_device = <MetalBackend as Backend>::Device::default();
    let tch_device = <TchBackend as Backend>::Device::default();
    
    // Test deep CNN
    println!("{}", "=".repeat(50));
    println!("TEST 1: Deep CNN (20 layers)");
    println!("{}", "=".repeat(50));
    
    let cnn_metal_diffs = test_deep_cnn::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device, 20
    );
    let cnn_ndarray_diffs = test_deep_cnn::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device, 20
    );
    
    analyze_error_growth(&cnn_metal_diffs, "CNN (Metal)");
    analyze_error_growth(&cnn_ndarray_diffs, "CNN (Ndarray)");
    
    // Test ResNet-like architecture
    println!("\n{}", "=".repeat(50));
    println!("TEST 2: ResNet-like (10 residual blocks = ~20 layers)");
    println!("{}", "=".repeat(50));
    
    let resnet_metal_diffs = test_deep_residual::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device, 10
    );
    let resnet_ndarray_diffs = test_deep_residual::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device, 10
    );
    
    analyze_error_growth(&resnet_metal_diffs, "ResNet (Metal)");
    analyze_error_growth(&resnet_ndarray_diffs, "ResNet (Ndarray)");
    
    // Test deep MLP
    println!("\n{}", "=".repeat(50));
    println!("TEST 3: Deep MLP (30 layers)");
    println!("{}", "=".repeat(50));
    
    let mlp_metal_diffs = test_deep_mlp::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device, 30
    );
    let mlp_ndarray_diffs = test_deep_mlp::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device, 30
    );
    
    analyze_error_growth(&mlp_metal_diffs, "MLP (Metal)");
    analyze_error_growth(&mlp_ndarray_diffs, "MLP (Ndarray)");
    
    // Test attention stack
    println!("\n{}", "=".repeat(50));
    println!("TEST 4: Attention Stack (12 layers - like BERT)");
    println!("{}", "=".repeat(50));
    
    let attn_metal_diffs = test_deep_attention::<TchBackend, MetalBackend>(
        "tch", &tch_device, "metal", &metal_device, 12
    );
    let attn_ndarray_diffs = test_deep_attention::<TchBackend, NdarrayBackend>(
        "tch", &tch_device, "ndarray", &ndarray_device, 12
    );
    
    analyze_error_growth(&attn_metal_diffs, "Attention (Metal)");
    analyze_error_growth(&attn_ndarray_diffs, "Attention (Ndarray)");
    
    // Final summary
    println!("\n{}", "=".repeat(50));
    println!("FINAL SUMMARY");
    println!("{}", "=".repeat(50));
    
    let threshold = 0.01; // 1% error threshold
    
    println!("\nFinal errors after full depth:");
    println!("Architecture    | Layers | Metal Error  | Ndarray Error | Status");
    println!("----------------|--------|--------------|---------------|--------");
    
    let cnn_metal_final = cnn_metal_diffs.last().unwrap_or(&0.0);
    let cnn_ndarray_final = cnn_ndarray_diffs.last().unwrap_or(&0.0);
    println!("CNN             | 20     | {:.8} | {:.8}   | {}", 
             cnn_metal_final, cnn_ndarray_final,
             if *cnn_metal_final > threshold || *cnn_ndarray_final > threshold { "❌" } else { "✅" });
    
    let resnet_metal_final = resnet_metal_diffs.last().unwrap_or(&0.0);
    let resnet_ndarray_final = resnet_ndarray_diffs.last().unwrap_or(&0.0);
    println!("ResNet          | ~20    | {:.8} | {:.8}   | {}", 
             resnet_metal_final, resnet_ndarray_final,
             if *resnet_metal_final > threshold || *resnet_ndarray_final > threshold { "❌" } else { "✅" });
    
    let mlp_metal_final = mlp_metal_diffs.last().unwrap_or(&0.0);
    let mlp_ndarray_final = mlp_ndarray_diffs.last().unwrap_or(&0.0);
    println!("MLP             | 30     | {:.8} | {:.8}   | {}", 
             mlp_metal_final, mlp_ndarray_final,
             if *mlp_metal_final > threshold || *mlp_ndarray_final > threshold { "❌" } else { "✅" });
    
    let attn_metal_final = attn_metal_diffs.last().unwrap_or(&0.0);
    let attn_ndarray_final = attn_ndarray_diffs.last().unwrap_or(&0.0);
    println!("Attention       | 12     | {:.8} | {:.8}   | {}", 
             attn_metal_final, attn_ndarray_final,
             if *attn_metal_final > threshold || *attn_ndarray_final > threshold { "❌" } else { "✅" });
    
    println!("\n⚠️ Error threshold for failure: {:.2}%", threshold * 100.0);
    
    // Check which architectures are most sensitive
    println!("\n🎯 Sensitivity Analysis:");
    let architectures = vec![
        ("CNN", *cnn_metal_final, *cnn_ndarray_final),
        ("ResNet", *resnet_metal_final, *resnet_ndarray_final),
        ("MLP", *mlp_metal_final, *mlp_ndarray_final),
        ("Attention", *attn_metal_final, *attn_ndarray_final),
    ];
    
    let most_sensitive = architectures.iter()
        .max_by(|a, b| {
            let max_a = a.1.max(a.2);
            let max_b = b.1.max(b.2);
            max_a.partial_cmp(&max_b).unwrap()
        })
        .unwrap();
    
    println!("  Most error-prone: {} (max error: {:.8})", 
             most_sensitive.0, most_sensitive.1.max(most_sensitive.2));
    
    let most_stable = architectures.iter()
        .min_by(|a, b| {
            let max_a = a.1.max(a.2);
            let max_b = b.1.max(b.2);
            max_a.partial_cmp(&max_b).unwrap()
        })
        .unwrap();
    
    println!("  Most stable:      {} (max error: {:.8})", 
             most_stable.0, most_stable.1.max(most_stable.2));
    
    println!("\n========================================");
}