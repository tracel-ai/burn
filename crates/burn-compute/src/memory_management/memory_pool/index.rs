use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::hash::Hash;
use hashbrown::HashMap;

/// Data Structure that helps to search items by size efficiently.
pub struct SearchIndex<T> {
    items_per_size: BTreeMap<usize, Vec<T>>,
    sizes_per_item: HashMap<T, usize>,
}

impl<T: PartialEq + Eq + Hash + Clone> SearchIndex<T> {
    /// Create a new item search index.
    pub fn new() -> Self {
        Self {
            items_per_size: BTreeMap::new(),
            sizes_per_item: HashMap::new(),
        }
    }

    /// Insert a new sized item into the search index.
    pub fn insert(&mut self, item: T, size: usize) {
        self.remove(&item);

        if let Some(values) = self.items_per_size.get_mut(&size) {
            values.push(item.clone())
        } else {
            self.items_per_size.insert(size, vec![item.clone()]);
        }
        self.sizes_per_item.insert(item, size);
    }

    /// Find the item by size range.
    pub fn find_by_size(
        &self,
        range: core::ops::Range<usize>,
    ) -> impl DoubleEndedIterator<Item = &T> {
        self.items_per_size.range(range).flat_map(|a| a.1)
    }

    /// Remove an item from the index.
    pub fn remove(&mut self, item: &T) {
        let size = match self.sizes_per_item.remove(item) {
            Some(size) => size,
            None => return,
        };

        if let Some(values) = self.items_per_size.get_mut(&size) {
            let mut removed_index = None;

            for (i, v) in values.iter().enumerate() {
                if v == item {
                    removed_index = Some(i);
                    break;
                }
            }

            if let Some(index) = removed_index {
                values.remove(index);
            }
        }
    }
}
