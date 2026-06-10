use std::sync::Arc;

use burn_core::{
    self as burn,
    module::{Module, ParamId},
    record::Record,
};

/// A [record](Record) for a [ParamId].
#[derive(Record, Clone)]
pub struct ParamIdRecord {
    /// The id.
    pub value: u64,
}

impl From<ParamId> for ParamIdRecord {
    fn from(value: ParamId) -> Self {
        ParamIdRecord { value: value.val() }
    }
}

/// Parameter grouping (adressed by parameter name).
pub type NamedParamGroup<T> = ParamGroupInner<T, String>;
/// Parameter grouping (adressed by parameter id).
pub type ParamGroup<T> = ParamGroupInner<T, ParamId>;
/// A [record](Record) for a [ParamGroup].
pub type ParamGroupRecord<T> = ParamGroupInnerRecord<T, ParamIdRecord>;

#[derive(new, Clone)]
/// Parameter grouping.
pub struct ParamGroupInner<T, I> {
    /// The tag of the group.
    pub tag: String,
    /// The list of parameters in this grou.
    pub params: Vec<I>,
    /// The group config
    pub config: T,
}

pub struct ParamGroup2 {
    matcher: ParamGroupMatcher,
}

impl ParamGroup2 {
    fn all_from_module<M: Module>(module: M) -> Self {
        // TODO: ParamGroupMatcher::Explicit
        // visitor
        todo!()
    }
}

enum ParamGroupMatcher {
    All,
    Explicit(Arc<Vec<ParamId>>),
    Combined(Arc<Vec<Self>>),
    Path(PathMatcher),
}

enum PathMatcher {
    Exact(String),
    // TODO: actually do regex
    Regex(String),
    Include(String),
}

#[derive(new, Clone, Record)]
/// Parameter grouping.
pub struct ParamGroupInnerRecord<T, I>
where
    T: Record,
    I: Record,
{
    /// The tag of the group.
    pub tag: String,
    /// The list of parameters in this grou.
    pub params: Vec<I>,
    /// The group config
    pub config: T,
}
