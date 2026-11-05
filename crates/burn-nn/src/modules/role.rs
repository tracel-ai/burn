//! The role a parent gives a child its type does not name.

use burn_core as burn;

use burn::instrument::compute;

/// What a component is for in its parent, where its type does not say: the query projection of
/// an attention is a [`Linear`](crate::Linear) like its key projection, and components merge on
/// their label among siblings. Running a child under its role opens an unpriced component of
/// that name around it, so the child keeps its own label and floor beneath.
pub(crate) struct Role(pub(crate) &'static str);

impl Role {
    /// An attention's projection of its query.
    pub(crate) const QUERY: Role = Role("query");
    /// An attention's projection of its key.
    pub(crate) const KEY: Role = Role("key");
    /// An attention's projection of its value.
    pub(crate) const VALUE: Role = Role("value");
    /// An attention's projection of its context back to the model's width.
    pub(crate) const OUTPUT: Role = Role("output");

    /// A feed-forward's projection into its hidden width.
    pub(crate) const INNER: Role = Role("inner");
    /// A feed-forward's projection back out of it.
    pub(crate) const OUTER: Role = Role("outer");

    /// A decoder layer's attention over the target.
    pub(crate) const SELF_ATTENTION: Role = Role("self attention");
    /// A decoder layer's attention from the target over the memory.
    pub(crate) const CROSS_ATTENTION: Role = Role("cross attention");

    /// The norm an encoder layer's attention sublayer is normalized by.
    pub(crate) const ATTENTION_NORM: Role = Role("attention norm");
    /// The norm a decoder layer's self-attention sublayer is normalized by.
    pub(crate) const SELF_ATTENTION_NORM: Role = Role("self attention norm");
    /// The norm a decoder layer's cross-attention sublayer is normalized by.
    pub(crate) const CROSS_ATTENTION_NORM: Role = Role("cross attention norm");
    /// The norm a layer's feed-forward sublayer is normalized by.
    pub(crate) const FEED_FORWARD_NORM: Role = Role("feed forward norm");

    /// A recurrent gate's projection of the step's input.
    pub(crate) const INPUT_TRANSFORM: Role = Role("input transform");
    /// A recurrent gate's projection of the hidden state.
    pub(crate) const HIDDEN_TRANSFORM: Role = Role("hidden transform");

    /// An LSTM's gate on what of the step enters the cell.
    pub(crate) const INPUT_GATE: Role = Role("input gate");
    /// An LSTM's gate on what of the cell is kept.
    pub(crate) const FORGET_GATE: Role = Role("forget gate");
    /// An LSTM's gate on what of the cell is output.
    pub(crate) const OUTPUT_GATE: Role = Role("output gate");
    /// An LSTM's candidate cell values.
    pub(crate) const CELL_GATE: Role = Role("cell gate");

    /// A GRU's gate between the previous hidden state and the candidate.
    pub(crate) const UPDATE_GATE: Role = Role("update gate");
    /// A GRU's gate on what of the previous hidden state the candidate sees.
    pub(crate) const RESET_GATE: Role = Role("reset gate");
    /// A GRU's candidate hidden state.
    pub(crate) const NEW_GATE: Role = Role("new gate");

    /// A bidirectional layer's pass from the first step to the last.
    pub(crate) const FORWARD: Role = Role("forward");
    /// A bidirectional layer's pass from the last step to the first.
    pub(crate) const REVERSE: Role = Role("reverse");

    /// Run `component` under this role.
    #[compute(label = self.0)]
    pub(crate) fn run<T>(&self, component: impl FnOnce() -> T) -> T {
        component()
    }
}
