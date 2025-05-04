use std::sync::Arc;

use crate::DatasetIterator;

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I>: Send + Sync {
    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<I>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

/// Format for saving labels
#[derive(Debug, Clone, Copy)]
pub enum LabelFormat {
    /// Text format with one label per line
    Txt,
    /// JSON format with an array of labels
    Json,
    /// YAML format with an array of labels
    Yaml,
}

/// The labeled dataset trait defines a dataset that contains labeled data.
/// It extends the basic Dataset trait with functionality to handle labels.
pub trait LabeledDataset<I, L>: Dataset<I>
where
    L: std::fmt::Display + serde::Serialize,
{
    /// Gets the label for the item at the given index.
    fn get_label(&self, index: usize) -> Option<L>;

    /// Gets all labels in the dataset.
    fn get_labels(&self) -> Vec<L> {
        (0..self.len()).filter_map(|i| self.get_label(i)).collect()
    }

    /// Saves the labels to a file in the default format (txt).
    fn save_labels<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        self.save_labels_with_format(path, LabelFormat::Txt)
    }

    /// Saves the labels to a file with the specified format.
    fn save_labels_with_format<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        format: LabelFormat,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let labels = self.get_labels();
        let mut file = std::fs::File::create(path)?;

        match format {
            LabelFormat::Txt => {
                for label in labels {
                    writeln!(file, "{}", label)?;
                }
            }
            LabelFormat::Json => {
                let json = serde_json::to_string_pretty(&labels)?;
                write!(file, "{}", json)?;
            }
            LabelFormat::Yaml => {
                let yaml = serde_yml::to_string(&labels).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                write!(file, "{}", yaml)?;
            }
        }

        Ok(())
    }
}

impl<D, I> Dataset<I> for Arc<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Arc<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I> Dataset<I> for Box<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Box<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

/// Utility function to write labels to a file
pub fn write_labels_to_file<P: AsRef<std::path::Path>, L: std::fmt::Display>(
    path: P,
    labels: &[L],
) -> std::io::Result<()> {
    let content = labels
        .iter()
        .map(|label| format!("{}", label))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, content)
}
