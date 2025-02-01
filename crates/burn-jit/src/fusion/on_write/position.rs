use super::ir::ElemwisePrecision;
use std::collections::BTreeMap;

/// Group output position by [element precision](ElemwisePrecision).
#[derive(Default, Debug)]
pub struct PositionMapper {
    map: BTreeMap<ElemwisePrecision, Vec<usize>>,
}

impl PositionMapper {
    /// Register a new output with the given precision and position.
    pub fn register(&mut self, precision: ElemwisePrecision, pos_handle: usize) {
        if let Some(positions) = self.map.get_mut(&precision) {
            positions.push(pos_handle);
        } else {
            self.map.insert(precision, vec![pos_handle]);
        }
    }

    /// Returns the right position from the precision and the global position in all outputs.
    pub fn resolve_index(&mut self, precision: &ElemwisePrecision, pos_handle: usize) -> u32 {
        self.map
            .get(&precision)
            .unwrap()
            .iter()
            .enumerate()
            .find(|(_pos_elem, pos_all)| **pos_all == pos_handle)
            .map(|(pos_elem, _pos_all)| pos_elem)
            .unwrap() as u32
    }
}
