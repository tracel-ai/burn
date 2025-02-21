use clap::{Parser, ValueEnum};
use mnist_no_std::{inference::no_std_world, proto::*};

#[derive(Clone, ValueEnum, Debug)]
enum InputType {
    /// The input file is a binary(must be 784 bytes)
    Binary,
    /// The input file is an image(must with dimension of 28x28)
    Image,
}

/// Loads a model from the given path, tests it with a given binary, and prints
/// the inference result.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The path of the model.
    #[arg(short, long, default_value = "model.bin")]
    model: String,
    /// The type of the input file
    #[arg(short, long, value_enum, default_value_t = InputType::Binary)]
    r#type: InputType,
    /// The path of the input file.
    #[arg(short, long)]
    input: String,
}

fn read_image_as_binary(args: &Args) -> Vec<u8> {
    let path = std::path::absolute(&args.input).unwrap();
    match args.r#type {
        InputType::Binary => {
            println!("Load binary from \"{}\"", path.display());
            std::fs::read(path).unwrap()
        }
        InputType::Image => {
            println!("Load image from \"{}\"", path.display());
            let img = image::open(&path)
                .unwrap()
                .resize_exact(
                    MNIST_IMAGE_WIDTH as u32,
                    MNIST_IMAGE_HEIGHT as u32,
                    image::imageops::FilterType::Nearest,
                )
                .into_luma8();
            img.to_vec()
        }
    }
}

fn main() {
    let args = Args::parse();

    let model_path = std::path::absolute(&args.model).unwrap();
    println!("Load model from \"{}\"", model_path.display());
    let record = std::fs::read(&model_path).unwrap();
    no_std_world::initialize(&record);

    let binary = read_image_as_binary(&args);
    assert_eq!(binary.len(), MNIST_IMAGE_SIZE);
    let result = no_std_world::infer(&binary);
    println!("Inference result is: {}", result);
}
