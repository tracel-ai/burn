use burn::data::dataset::Dataset;
use custom_csv_dataset::dataset::DiabetesDataset;

fn main() {
    // TODO:
    // - Should the CSV reader be added to parametrize `InMemDataset::from_csv()`? Current implementation does not allow a wide configuration for the csv to read (but maybe this is better like this?)
    // -> from_csv(path, ReaderBuilder)

    // NEXT:
    // - Huggingface source

    // - Add another example for images? (similar to `torchvision.datasets.DatasetFolder`)
    // - Add batcher to complete the example (new)

    // let source = env::args().skip(1).next();
    let dataset = DiabetesDataset::new().expect("Could not load diabetes dataset");

    println!("Dataset loaded with {} rows", dataset.len());

    let item = dataset.get(0).unwrap();
    println!("First item:\n{:?}", item);

    let item = dataset.get(441).unwrap();
    println!("Last item:\n{:?}", item);
}
