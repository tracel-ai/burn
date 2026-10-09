# Burn Dataset

> Datasets, transformations and data sources for [Burn](https://github.com/tracel-ai/burn)

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-dataset.svg)](https://crates.io/crates/burn-dataset)
[![Documentation](https://docs.rs/burn-dataset/badge.svg)](https://docs.rs/burn-dataset)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Applications enable the `dataset` feature of `burn` (also enabled by `train`) and use this crate as
`burn::data::dataset`.

- `Dataset` is random access by index. `InMemDataset` holds items in memory, and the `sqlite`
  feature adds `SqliteDataset` for datasets larger than memory.
- `transform`: mapping, sampling, shuffling, windowing, partial views, selection and
  composition of datasets, applied lazily.
- `source`: dataset downloads, including Hugging Face datasets.
- `vision`, `nlp`, `audio`: ready-made datasets such as MNIST, CIFAR, image folders, AG News and
  Speech Commands, behind the features of the same names.

See the [dataset chapter](https://burn.dev/books/burn/building-blocks/dataset.html) of the Burn
Book.

## Feature Flags

- `sqlite`: SQLite-backed datasets, using the [Turso](https://turso.tech/) engine. `sqlite-bundled`
  is a deprecated alias.
- `vision`, `nlp`, `audio`: domain datasets and loaders. `builtin-sources` enables the downloadable
  vision and NLP datasets.
- `dataframe`: datasets backed by a Polars `DataFrame`.
- `network`: file downloads.
- `fake`: generated datasets for tests.
- `tracing`: instrument operations with the `tracing` crate.

Try the audio dataset with:

```shell
cargo run -p burn-dataset --example speech_commands --features audio
```

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
