/// A set of nodes of a [Dag](super::Dag), identified by their node index.
///
/// The first 64 node indices live in an inline word, so subgraphs over small graphs — the common
/// case, given the block cap — never touch the heap. Larger graphs spill into a word vector kept
/// canonical (no trailing zero words), which makes the derived equality structural.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubGraph {
    /// Nodes `0..64`.
    word0: u64,
    /// Nodes `64..`: word `i` holds nodes `(i + 1) * 64..(i + 2) * 64`.
    spill: Vec<u64>,
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
        self.word0 == 0 && self.spill.is_empty()
    }

    /// The number of nodes in the subgraph.
    pub fn len(&self) -> usize {
        let spill: u32 = self.spill.iter().map(|word| word.count_ones()).sum();
        (self.word0.count_ones() + spill) as usize
    }

    /// Add a node to the subgraph.
    pub fn insert(&mut self, node: usize) {
        if node < BITS {
            self.word0 |= 1 << node;
            return;
        }
        let word = node / BITS - 1;
        if word >= self.spill.len() {
            self.spill.resize(word + 1, 0);
        }
        self.spill[word] |= 1 << (node % BITS);
    }

    /// Whether the subgraph contains the node.
    #[cfg(test)]
    pub fn contains(&self, node: usize) -> bool {
        if node < BITS {
            return self.word0 & (1 << node) != 0;
        }
        self.spill
            .get(node / BITS - 1)
            .is_some_and(|word| word & (1 << (node % BITS)) != 0)
    }

    /// Add every node of the other subgraph.
    pub fn union_with(&mut self, other: &Self) {
        self.word0 |= other.word0;
        if other.spill.len() > self.spill.len() {
            self.spill.resize(other.spill.len(), 0);
        }
        for (word, other) in self.spill.iter_mut().zip(&other.spill) {
            *word |= other;
        }
    }

    /// Keep only the nodes also contained in the other subgraph.
    pub fn intersect_with(&mut self, other: &Self) {
        self.word0 &= other.word0;
        self.spill.truncate(other.spill.len());
        for (word, other) in self.spill.iter_mut().zip(&other.spill) {
            *word &= other;
        }
        self.trim();
    }

    /// Remove every node of the other subgraph.
    pub fn subtract(&mut self, other: &Self) {
        self.word0 &= !other.word0;
        for (word, other) in self.spill.iter_mut().zip(&other.spill) {
            *word &= !other;
        }
        self.trim();
    }

    /// Whether the two subgraphs share at least one node.
    #[cfg(test)]
    pub fn intersects(&self, other: &Self) -> bool {
        self.word0 & other.word0 != 0
            || self.spill.iter().zip(&other.spill).any(|(a, b)| a & b != 0)
    }

    /// The nodes of the subgraph, in ascending index order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        core::iter::once(self.word0)
            .chain(self.spill.iter().copied())
            .enumerate()
            .flat_map(|(i, word)| {
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

    /// Restore the canonical form (no trailing zero spill words).
    fn trim(&mut self) {
        while self.spill.last() == Some(&0) {
            self.spill.pop();
        }
    }
}
