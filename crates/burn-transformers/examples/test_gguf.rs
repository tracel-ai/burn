use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::Device;
use burn_transformers::gguf::{GGUFArchitecture, GGUFModel};
use burn_transformers::tokenizer::gguf::GGUFTokenizer;
use burn_transformers::tokenizer::Tokenizer;
use clap::Parser;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "Test GGUF model loading and tokenization")]
struct Args {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Test text to tokenize
    #[arg(short, long, default_value = "Hello, world! This is a test.")]
    test_text: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    println!("Loading GGUF model from: {}", args.model_path.display());
    let start = Instant::now();

    // Load the GGUF model
    let model = GGUFModel::load(&args.model_path)?;
    let load_time = start.elapsed();

    println!("Model loaded in {:.2?}", load_time);
    println!("Model has {} tensors", model.tensors.len());

    // Print architecture information
    println!("\nModel architecture: {:?}", model.architecture);

    // Print some metadata
    println!("\nModel metadata:");
    if let Some(name) = model.get_metadata("general.name") {
        println!("  Name: {:?}", name);
    }
    if let Some(family) = model.get_metadata("general.family") {
        println!("  Family: {:?}", family);
    }
    if let Some(architecture) = model.get_metadata("general.architecture") {
        println!("  Architecture: {:?}", architecture);
    }
    if let Some(quantization_version) = model.get_metadata("general.quantization_version") {
        println!("  Quantization version: {:?}", quantization_version);
    }

    // Print model architecture-specific configuration
    println!("\nModel config:");
    let arch_str = model.architecture.as_str();
    if model.architecture != GGUFArchitecture::Unknown {
        if let Some(context_len) = model.get_metadata(&format!("{}.context_length", arch_str)) {
            println!("  Context length: {:?}", context_len);
        }
        if let Some(embedding_len) = model.get_metadata(&format!("{}.embedding_length", arch_str)) {
            println!("  Embedding length: {:?}", embedding_len);
        }
        if let Some(head_count) = model.get_metadata(&format!("{}.attention.head_count", arch_str))
        {
            println!("  Attention head count: {:?}", head_count);
        }
        if let Some(head_count_kv) =
            model.get_metadata(&format!("{}.attention.head_count_kv", arch_str))
        {
            println!("  KV head count: {:?}", head_count_kv);
        }
        if let Some(block_count) = model.get_metadata(&format!("{}.block_count", arch_str)) {
            println!("  Block count: {:?}", block_count);
        }
    }

    // Print information about all tensors
    println!("\nTensor information:");
    for tensor in &model.tensors {
        println!(
            "  {}: Shape {:?}, Type {}",
            tensor.name, tensor.dimensions, tensor.tensor_type
        );
    }

    // Try to load a few tensors to test our implementation
    println!("\nTesting tensor loading...");

    // Using CPU device for testing
    use burn::backend::NdArray;
    let device = Device::<NdArray>::Cpu;

    // Find a couple of tensors to test
    let test_tensors = vec![
        // Find a couple of key tensors (without knowing the exact tensor names)
        &model.tensors[0],                       // First tensor
        &model.tensors[model.tensors.len() / 2], // Middle tensor
        &model.tensors[model.tensors.len() - 1], // Last tensor
    ];

    for tensor_info in test_tensors {
        let name = &tensor_info.name;

        // Determine dimensionality and load accordingly
        let dim_count = tensor_info.dimensions.len();
        match dim_count {
            1 => {
                let start = Instant::now();
                let tensor_result = model.get_tensor::<NdArray, 1>(name, &device);
                let load_time = start.elapsed();

                match tensor_result {
                    Ok(tensor) => println!(
                        "  Successfully loaded 1D tensor '{}' with shape {:?} in {:.2?}",
                        name,
                        tensor.shape(),
                        load_time
                    ),
                    Err(e) => println!("  Failed to load 1D tensor '{}': {}", name, e),
                }
            }
            2 => {
                let start = Instant::now();
                let tensor_result = model.get_tensor::<NdArray, 2>(name, &device);
                let load_time = start.elapsed();

                match tensor_result {
                    Ok(tensor) => println!(
                        "  Successfully loaded 2D tensor '{}' with shape {:?} in {:.2?}",
                        name,
                        tensor.shape(),
                        load_time
                    ),
                    Err(e) => println!("  Failed to load 2D tensor '{}': {}", name, e),
                }
            }
            _ => println!(
                "  Skipping tensor '{}' with unsupported dimensionality {}",
                name, dim_count
            ),
        }
    }

    // Try to get vocabulary
    println!("\nTesting vocabulary retrieval...");
    match model.get_vocabulary() {
        Some(vocab) => {
            let vocab_size = vocab.len();
            println!("  Retrieved vocabulary with {} tokens", vocab_size);
            if vocab_size > 0 {
                println!("  First few tokens:");
                for (i, (token, score)) in vocab.iter().take(5).enumerate() {
                    // Display token in a readable format, handling control characters
                    let display_token = token
                        .chars()
                        .map(|c| if c.is_control() { '□' } else { c })
                        .collect::<String>();
                    println!("    {}: \"{}\" (score: {})", i, display_token, score);
                }
                let last_idx = vocab.len() - 1;
                let (last_token, last_score) = &vocab[last_idx];
                let display_last = last_token
                    .chars()
                    .map(|c| if c.is_control() { '□' } else { c })
                    .collect::<String>();
                println!("    ...");
                println!(
                    "    {}: \"{}\" (score: {})",
                    last_idx, display_last, last_score
                );
            }
        }
        None => println!("  No vocabulary found in the model"),
    }

    // Try to get typed vocabulary (with token types)
    println!("\nTesting typed vocabulary retrieval...");
    match model.get_typed_vocabulary() {
        Some(typed_vocab) => {
            let vocab_size = typed_vocab.len();
            println!("  Retrieved typed vocabulary with {} tokens", vocab_size);
            if vocab_size > 0 {
                println!("  First few tokens (with types):");
                for (i, entry) in typed_vocab.iter().take(5).enumerate() {
                    // Display token in a readable format, handling control characters
                    let display_token = entry
                        .token
                        .chars()
                        .map(|c| if c.is_control() { '□' } else { c })
                        .collect::<String>();
                    println!(
                        "    {}: \"{}\" (score: {}, type: {})",
                        i, display_token, entry.score, entry.token_type
                    );
                }

                // Show breakdown of token types
                let normal_tokens = typed_vocab.iter().filter(|e| e.token_type == 0).count();
                let special_tokens = typed_vocab.iter().filter(|e| e.token_type == 1).count();
                let other_tokens = typed_vocab.iter().filter(|e| e.token_type > 1).count();

                println!("  Token type breakdown:");
                println!("    Normal tokens (type 0): {}", normal_tokens);
                println!("    Special tokens (type 1): {}", special_tokens);
                if other_tokens > 0 {
                    println!("    Other token types (>1): {}", other_tokens);
                }
            }
        }
        None => println!("  No typed vocabulary found in the model"),
    }

    // Test the tokenizer
    println!("\nTesting tokenizer...");
    let start = Instant::now();
    let tokenizer_result = GGUFTokenizer::new(args.model_path.to_str().unwrap());
    let tokenizer_load_time = start.elapsed();

    match tokenizer_result {
        Ok(tokenizer) => {
            println!("  Tokenizer loaded in {:.2?}", tokenizer_load_time);
            println!("  Vocabulary size: {}", tokenizer.vocab_size());

            // Test encoding and decoding
            let text = &args.test_text;
            println!("  Testing with text: \"{}\"", text);

            // Without special tokens
            let start = Instant::now();
            let tokens = tokenizer.encode(text, false, false);
            let encode_time = start.elapsed();

            println!(
                "  Encoded {} tokens in {:.2?}: {:?}",
                tokens.len(),
                encode_time,
                tokens
            );

            let start = Instant::now();
            match tokenizer.decode(tokens.clone()) {
                Ok(decoded) => {
                    let decode_time = start.elapsed();
                    println!("  Decoded in {:.2?}: \"{}\"", decode_time, decoded);
                }
                Err(e) => println!("  Failed to decode: {}", e),
            }

            // With special tokens
            let tokens_with_special = tokenizer.encode(text, true, true);
            println!(
                "  With special tokens: {} tokens: {:?}",
                tokens_with_special.len(),
                tokens_with_special
            );

            match tokenizer.decode(tokens_with_special) {
                Ok(decoded) => println!("  Decoded with special tokens: \"{}\"", decoded),
                Err(e) => println!("  Failed to decode with special tokens: {}", e),
            }
        }
        Err(e) => println!("  Failed to load tokenizer: {}", e),
    }

    println!("\nGGUF model testing completed successfully!");
    Ok(())
}
