#[derive(Default, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    value: String,
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.value)
    }
}

impl NodeId {
    pub fn as_str(&self) -> &str {
        self.value.as_str()
    }
}
