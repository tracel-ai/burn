# Dataset

In most deep learning training performed on datasets (with perhaps the exception of reinforcement learning), it is
essential to provide a convenient and performant API.
The dataset trait is quite similar to the dataset abstract class in PyTorch:

```rust, ignore
pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
}
```

The dataset trait assumes a fixed-length set of items that can be randomly accessed in constant
time. This is a major difference from datasets that use Apache Arrow underneath to improve streaming
performance. Datasets in Burn don't assume _how_ they are going to be accessed; it's just a
collection of items.

However, you can compose multiple dataset transformations to lazily obtain what you want with zero
pre-processing, so that your training can start instantly!

## Transformation

Transformations in Burn are all lazy and modify one or multiple input datasets. The goal of these
transformations is to provide you with the necessary tools so that you can model complex data
distributions.

| Transformation    | Description                                                                                                              |
|-------------------|--------------------------------------------------------------------------------------------------------------------------|
| `SamplerDataset`  | Samples items from a dataset. This is a convenient way to model a dataset as a probability distribution of a fixed size. |
| `ShuffledDataset` | Maps each input index to a random index, similar to a dataset sampled without replacement.                               |
| `PartialDataset`  | Returns a view of the input dataset with a specified range.                                                              |
| `MapperDataset`   | Computes a transformation lazily on the input dataset.                                                                   |
| `ComposedDataset` | Composes multiple datasets together to create a larger one without copying any data.                                     |
| `WindowDataset`   | Dataset designed to work with overlapping windows of data extracted from an input dataset.                               |

Let us look at the basic usages of each dataset transform and how they can be composed together. These transforms
are lazy by default except when specified, reducing the need for unnecessary intermediate allocations and improving
performance. The full documentation of each transform can be found at
the [API reference](https://burn.dev/docs/burn/data/dataset/transform/index.html).

* **SamplerDataset**: This transform can be used to sample items from a dataset with (default) or without replacement.
  Transform is initialized with a sampling size which can be bigger or smaller than the input dataset size. This is
  particularly useful in cases where we want to checkpoint larger datasets more often during training
  and smaller datasets less often as the size of an epoch is now controlled by the sampling size. Sample usage:

```rust, ignore
type DbPedia = SqliteDataset<DbPediaItem>;
let dataset: DbPedia = HuggingfaceDatasetLoader::new("dbpedia_14")
        .dataset("train").
        .unwrap();
                
let dataset = SamplerDataset<DbPedia, DbPediaItem>::new(dataset, 10000);
```

* **ShuffledDataset**: This transform can be used to shuffle the items of a dataset. Particularly useful before
  splitting
  the raw dataset into train/test splits. Can be initialized with a seed to ensure reproducibility.

```rust, ignore
let dataset = ShuffledDataset<DbPedia, DbPediaItem>::with_seed(dataset, 42);
```

* **PartialDataset**: This transform is useful to return a view of the dataset with specified start and end indices.
  Used
  to create train/val/test splits. In the example below, we show how to chain ShuffledDataset and PartialDataset to
  create
  splits.

```rust, ignore
// define chained dataset type here for brevity
type PartialData = PartialDataset<ShuffledDataset<DbPedia, DbPediaItem>>;
let len = dataset.len();
let split == "train"; // or "val"/"test"

let data_split = match split {
            "train" => PartialData::new(dataset, 0, len * 8 / 10), // Get first 80% dataset
            "test" => PartialData::new(dataset, len * 8 / 10, len), // Take remaining 20%
            _ => panic!("Invalid split type"),                     // Handle unexpected split types
        };
```

* **MapperDataset**: This transform is useful to apply a transformation on each of the items of a dataset. Particularly
  useful for normalization of image data when channel means are known.

* **ComposedDataset**: This transform is useful to compose multiple datasets downloaded from multiple sources (say
  different HuggingfaceDatasetLoader sources) into a single bigger dataset which can be sampled from one source.

## Storage

There are multiple dataset storage options available for you to choose from. The choice of the
dataset to use should be based on the dataset's size as well as its intended purpose.

| Storage         | Description                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| `InMemDataset`  | In-memory dataset that uses a vector to store items. Well-suited for smaller datasets.                                    |
| `SqliteDataset` | Dataset that uses SQLite to index items that can be saved in a simple SQL database file. Well-suited for larger datasets. |

## Sources

For now, there is only one dataset source available with Burn, but more to come!

### Hugging Face

You can easily import any Hugging Face dataset with Burn. We use SQLite as the storage to avoid
downloading the model each time or starting a Python process. You need to know the format of each
item in the dataset beforehand. Here's an example with the
[dbpedia dataset](https://huggingface.co/datasets/dbpedia_14).

```rust, ignore
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    pub title: String,
    pub content: String,
    pub label: usize,
}

fn main() {
    let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
        .dataset("train") // The training split.
        .unwrap();
}
```

We see that items must derive `serde::Serialize`, `serde::Deserialize`, `Clone`, and `Debug`, but
those are the only requirements.

**What about streaming datasets?**

There is no streaming dataset API with Burn, and this is by design! The learner struct will iterate
multiple times over the dataset and only checkpoint when done. You can consider the length of the
dataset as the number of iterations before performing checkpointing and running the validation.
There is nothing stopping you from returning different items even when called with the same `index`
multiple times.
