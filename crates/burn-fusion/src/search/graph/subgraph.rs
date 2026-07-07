/// A set of nodes of a [Dag](super::Dag), identified by their node index.
///
/// Backed by a growable bitset, so it supports graphs of any size. The word vector is kept
/// canonical (no trailing zero words), which makes the derived equality structural.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubGraph {
    words: Vec<u64>,
}

const BITS: usize = u64::BITS as usize;

impl SubGraph {
    /// The empty subgraph.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The subgraph containing a single node.
    pub fn single(node: usize) -> Self {
        let mut set = Self::empty();
        set.insert(node);
        set
    }

    /// Whether the subgraph contains no node.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Add a node to the subgraph.
    pub fn insert(&mut self, node: usize) {
        let word = node / BITS;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << (node % BITS);
    }

    /// Whether the subgraph contains the node.
    pub fn contains(&self, node: usize) -> bool {
        self.words
            .get(node / BITS)
            .is_some_and(|word| word & (1 << (node % BITS)) != 0)
    }

    /// Add every node of the other subgraph.
    pub fn union_with(&mut self, other: &Self) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word |= other;
        }
    }

    /// Keep only the nodes also contained in the other subgraph.
    pub fn intersect_with(&mut self, other: &Self) {
        self.words.truncate(other.words.len());
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word &= other;
        }
        self.trim();
    }

    /// Remove every node of the other subgraph.
    pub fn subtract(&mut self, other: &Self) {
        for (word, other) in self.words.iter_mut().zip(&other.words) {
            *word &= !other;
        }
        self.trim();
    }

    /// Whether the two subgraphs share at least one node.
    #[cfg(test)]
    pub fn intersects(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(&other.words)
            .any(|(a, b)| a & b != 0)
    }

    /// The nodes of the subgraph, in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &word)| {
            let mut bits = word;
            core::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(i * BITS + bit)
            })
        })
    }

    /// Restore the canonical form (no trailing zero words).
    fn trim(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
}
