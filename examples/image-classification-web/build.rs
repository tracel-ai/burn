/// This build script generates the model code from the ONNX file and the labels from the text file.
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
};

use burn_import::onnx::ModelGen;

const LABEL_SOURCE_FILE: &str = "src/model/label.txt";
const LABEL_DEST_FILE: &str = "model/label.rs";
const INPUT_ONNX_FILE: &str = "src/model/squeezenet1_opset16.onnx";
const OUT_DIR: &str = "model/";

fn main() {
    // Re-run the build script if model files change.
    println!("cargo:rerun-if-changed=src/model");

    // Generate the model code from the ONNX file.
    // Model weights are embedded in the binary for WebAssembly compatibility.
    ModelGen::new()
        .input(INPUT_ONNX_FILE)
        .out_dir(OUT_DIR)
        .embed_states(true)
        .run_from_script();

    // Generate the labels from the synset.txt file.
    generate_labels_from_txt_file().unwrap();
}

/// Read labels from synset.txt and store them in a vector of strings in a Rust file.
fn generate_labels_from_txt_file() -> std::io::Result<()> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join(LABEL_DEST_FILE);
    let mut f = File::create(dest_path)?;

    let file = File::open(LABEL_SOURCE_FILE)?;
    let reader = BufReader::new(file);

    writeln!(f, "pub static LABELS: &[&str] = &[")?;
    for line in reader.lines() {
        writeln!(
            f,
            "    \"{}\",",
            extract_simple_label(line.unwrap()).unwrap()
        )?;
    }
    writeln!(f, "];")?;

    Ok(())
}

/// Extract the simple label from the full label.
///
/// The full label is of the form: "n01537544 indigo bunting, indigo finch, indigo bird, Passerina cyanea"
/// The simple label is of the form: "indigo bunting"
fn extract_simple_label(input: String) -> Option<String> {
    // Split the string based on the space character.
    let mut parts = input.split(' ');

    // Skip the first part (the alphanumeric code).
    parts.next()?;

    // Get the remaining string.
    let remaining = parts.collect::<Vec<&str>>().join(" ");

    // Find the first comma, if it exists, and take the substring before it.
    let end_index = remaining.find(',').unwrap_or(remaining.len());

    Some(remaining[0..end_index].to_string())
}
