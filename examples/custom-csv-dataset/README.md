# Custom CSV Dataset

This example demonstrates two ways to load a CSV dataset and implement the `Dataset` trait. For this example, we use the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (original [source](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)), which contains 442 patient records.

## InMemDataset

The [custom-csv-dataset](src/dataset.rs) example uses [`InMemDataset::from_csv(path)`](src/dataset.rs#L80) to read the csv dataset file into a vector (in-memory) of [`DiabetesPatient`](src/dataset.rs#L13) records (struct) with the help of `serde`.

### Example Usage

```sh
cargo run --example custom-csv-dataset
```

## DataframeDataset (Polars)

The [dataframe-dataset](src/dataframe_dataset.rs) example demonstrates using [`DataframeDataset`](src/dataframe_dataset.rs#L61) with [Polars](https://www.pola.rs/) as the backend. This approach is well-suited for efficient data manipulation and analysis of larger datasets.

The same diabetes dataset is loaded into a Polars DataFrame, which is then wrapped by `DataframeDataset` to implement the `Dataset` trait.

### Example Usage

```sh
cargo run --example dataframe-dataset --features dataframe
```