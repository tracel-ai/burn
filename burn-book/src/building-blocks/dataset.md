# Dataset

Most deep learning training being done on datasets –with perhaps the exception of reinforcement learning–, it is essential to provide a convenient and performant API.
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
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `SamplerDataset`  | Samples items from a dataset. This is a convenient way to model a dataset as a probability distribution of a fixed size. |
| `ShuffledDataset` | Maps each input index to a random index, similar to a dataset sampled without replacement.                               |
| `PartialDataset`  | Returns a view of the input dataset with a specified range.                                                              |
| `MapperDataset`   | Computes a transformation lazily on the input dataset.                                                                   |
| `ComposedDataset` | Composes multiple datasets together to create a larger one without copying any data.                                     |

## Storage

There are multiple dataset storage options available for you to choose from. The choice of the
dataset to use should be based on the dataset's size as well as its intended purpose.

| Storage         | Description                                                                                                               |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
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
