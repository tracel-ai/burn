use burn::data::dataset::Dataset;
use custom_csv_dataset::dataset::DiabetesDataset;

fn main() {
    let dataset = DiabetesDataset::new().expect("Could not load diabetes dataset");

    println!("Dataset loaded with {} rows", dataset.len());

    // Display first and last elements
    let item = dataset.get(0).unwrap();
    println!("First item:\n{:?}", item);

    let item = dataset.get(441).unwrap();
    println!("Last item:\n{:?}", item);
}
