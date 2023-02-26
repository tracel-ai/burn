/// Duplicate the struct for each entry in the array that is true.
pub fn duplicate<T: Clone, const N: usize>(entries: [bool; N], obj: T) -> [Option<T>; N] {
    entries.map(|entry| match entry {
        true => Some(obj.clone()),
        false => None,
    })
}
