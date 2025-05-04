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

/// The labeled dataset trait defines a dataset that contains labeled data.
/// It extends the basic Dataset trait with functionality to handle labels.
pub trait LabeledDataset<I, L>: Dataset<I> {
    /// Gets the label for the item at the given index.
    fn get_label(&self, index: usize) -> Option<L>;

    /// Gets all labels in the dataset.
    fn get_labels(&self) -> Vec<L> {
        (0..self.len()).filter_map(|i| self.get_label(i)).collect()
    }

    /// Saves the labels to a file.
    fn save_labels<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()>
    where
        L: std::fmt::Display,
    {
        let labels = self.get_labels();
        write_labels_to_file(path, &labels)
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
