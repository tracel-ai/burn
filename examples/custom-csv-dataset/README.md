# Custom CSV Dataset

The [custom-csv-dataset](src/dataset.rs) example implements the `Dataset` trait to retrieve dataset elements from a `.csv` file on disk. For this example, we use the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (original [source](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)).

The dataset only contains 442 records, so we use [`InMemDataset::from_csv(path)`](src/dataset.rs#L80) method to read the csv dataset file into a vector (in-memory) of [`DiabetesPatient`](src/dataset.rs#L13) records (struct) with the help of `serde`.

## Example Usage

```sh
cargo run --example custom-csv-dataset
```