# Custom CSV Dataset

The [custom-csv-dataset](src/dataset.rs) example implements the `Dataset` trait to retrieve dataset elements from a `.csv` file on disk. For this example, we use the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (original [source](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)).

While we could use the `InMemDataset::from_csv(path)` method to read the csv file to store the entire dataset elements to a vector (in-memory), the default `csv::Reader` uses the comma `,` as a delimiter. Instead, we implement the `DiabetesDataset::new(path)` method to parse the `.csv` file with `serde` into a vector of `DiabetesPatient` records (struct).

## Example Usage

```sh
cargo run --example custom-csv-dataset
```


<!-- ```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using ndarray backend"
cargo run --example mnist --release --features ndarray                # CPU NdArray Backend - f32 - single thread
cargo run --example mnist --release --features ndarray-blas-openblas  # CPU NdArray Backend - f32 - blas with openblas
cargo run --example mnist --release --features ndarray-blas-netlib    # CPU NdArray Backend - f32 - blas with netlib
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu113                                       # Set the cuda version
cargo run --example mnist --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example mnist --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using wgpu backend"
cargo run --example mnist --release --features wgpu
``` -->
