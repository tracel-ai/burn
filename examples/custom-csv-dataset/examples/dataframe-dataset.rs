use burn::data::dataset::Dataset;
use custom_csv_dataset::dataframe_dataset::DiabetesDataframeDataset;

fn main() {
    let dataset = DiabetesDataframeDataset::new()
        .expect("Could not load diabetes dataset with DataframeDataset");

    println!(
        "Dataset loaded with {} rows using DataframeDataset",
        dataset.len()
    );

    // Display first and last elements
    let item = dataset.get(0).unwrap();
    println!("First item:\n{item:?}");

    let item = dataset.get(441).unwrap();
    println!("Last item:\n{item:?}");
}
