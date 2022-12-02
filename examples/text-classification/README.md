# Text Classification

The example can be run like so:

```bash
git clone https://github.com/burn-rs/burn.git
cd burn
# Use the --release flag to really speed up training.
export TORCH_CUDA_VERSION=cu113                            # Set the cuda version
cargo run --example text-classification-ag-news --release  # Train on the ag news dataset
cargo run --example text-classification-db-pedia --release # Train on the db pedia dataset
```
