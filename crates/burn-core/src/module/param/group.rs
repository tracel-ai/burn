use std::sync::Arc;

use burn_std::id::ParamId;
use burn_tensor::{Bool, Int, Tensor};
use regex::Regex;

use crate::module::{Module, ModuleVisitor, Param};

/// Errors tied to [ParamGroup]'s.
#[derive(Debug)]
pub enum ParamGroupError {
    /// Parameters used to match a parameter are invalid.
    InvalidParameter(String),
    /// Error while creating a group.
    CreationError(String),
}

#[derive(Default)]
struct ParamIdCollector {
    ids: Vec<ParamId>,
}

impl ParamIdCollector {
    pub fn ids(&self) -> Vec<ParamId> {
        self.ids.clone()
    }
}

impl ModuleVisitor for ParamIdCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.ids.push(param.id);
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.ids.push(param.id);
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.ids.push(param.id);
    }
}

/// A way to represent a group of parameter for a Burn module.
pub struct ParamGroup {
    matcher: ParamGroupMatcher,
    excludes: Option<ParamGroupMatcher>,
}

impl ParamGroup {
    /// Evaluates whether a given parameter ID and its module path match this group.
    pub fn matches(&self, id: &ParamId, path: Option<&str>) -> Result<bool, ParamGroupError> {
        let matched = self.matcher.matches(id, path)?;

        let excluded = if let Some(exclude_matcher) = &self.excludes {
            exclude_matcher.matches(id, path)?
        } else {
            false
        };

        Ok(matched && !excluded)
    }

    /// Returns a parameter group with all of the module's parameters.
    pub fn ids_from_module<M: Module>(module: M) -> Self {
        let mut collector = ParamIdCollector::default();
        module.visit(&mut collector);
        Self {
            matcher: ParamGroupMatcher::from_ids(collector.ids()),
            excludes: None,
        }
    }

    /// Matches parameters by exact text path (e.g., "model.backbone.linear.weight")
    pub fn from_path(path: impl Into<String>) -> Self {
        ParamGroup::from_paths(vec![path])
    }

    /// Matches parameters by exact text paths (e.g., "model.backbone.linear.weight", etc.)
    pub fn from_paths(paths: Vec<impl Into<String>>) -> Self {
        Self {
            matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Exact(
                paths.into_iter().map(|p| p.into()).collect(),
            ))),
            excludes: None,
        }
    }

    /// Matches parameters including this string (e.g., "backbone")
    pub fn from_predicate(path: impl Into<String>) -> Self {
        ParamGroup::from_predicates(vec![path])
    }

    /// Matches parameters including all these string (e.g., "backbone" and "linear")
    pub fn from_predicates(paths: Vec<impl Into<String>>) -> Self {
        Self {
            matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Include(
                paths.into_iter().map(|p| p.into()).collect(),
            ))),
            excludes: None,
        }
    }

    /// Matches parameters by regex pattern (e.g., "^model\.layer\.\d+$")
    pub fn from_regex<S: AsRef<str>>(pattern: S) -> Result<Self, regex::Error> {
        ParamGroup::from_regexes(vec![pattern])
    }

    /// Matches parameters by regex patterns (e.g., "^model\.layer\.\d+$", etc.)
    pub fn from_regexes<I, S>(patterns: I) -> Result<Self, regex::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut new_patterns = vec![];
        for pattern in patterns {
            new_patterns.push(Regex::new(pattern.as_ref())?);
        }
        Ok(Self {
            matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Regex(new_patterns))),
            excludes: None,
        })
    }

    /// Matches any parameter.
    pub fn all() -> Self {
        Self {
            matcher: ParamGroupMatcher::All,
            excludes: None,
        }
    }

    /// Matches a specific slice of predefined parameter IDs.
    pub fn from_ids(ids: Vec<ParamId>) -> Self {
        Self {
            matcher: ParamGroupMatcher::Explicit(Arc::new(ids)),
            excludes: None,
        }
    }

    /// Fuse two parameter group.
    pub fn fuse(self, other: &Self) -> Self {
        Self {
            matcher: self.matcher.fuse(&other.matcher),
            excludes: None,
        }
    }

    // TODO: Should we ignore exclusions?
    /// Exclude an existing group from the current group. If the other group already has exclusions, they are ignored.
    pub fn exclude(&mut self, group: &Self) {
        self.excludes = match &self.excludes {
            Some(excluded) => Some(excluded.clone().fuse(&group.matcher)),
            None => Some(group.matcher.clone()),
        };
    }
}

#[derive(Clone)]
enum ParamGroupMatcher {
    All,
    Explicit(Arc<Vec<ParamId>>),
    Path(Arc<PathMatcher>),
    Combined(Arc<Vec<Self>>),
}

impl ParamGroupMatcher {
    pub fn from_ids(ids: Vec<ParamId>) -> Self {
        Self::Explicit(Arc::new(ids))
    }

    pub(crate) fn matches(
        &self,
        id: &ParamId,
        path: Option<&str>,
    ) -> Result<bool, ParamGroupError> {
        match self {
            Self::All => Ok(true),
            Self::Explicit(ids) => Ok(ids.contains(&id)),
            Self::Path(matcher) => path
                .ok_or_else(|| {
                    ParamGroupError::InvalidParameter(
                        "Matching on a path requires giving `matches` a path.".into(),
                    )
                })
                .map(|p| matcher.matches(p)),
            Self::Combined(matchers) => Ok(matchers.iter().try_fold(false, |acc, m| {
                m.matches(id, path).map(|matched| acc || matched)
            })?),
        }
    }

    fn push_combined(self, other: Self) -> Self {
        match self {
            ParamGroupMatcher::Combined(param_group_matchers) => match other.clone() {
                ParamGroupMatcher::All => Self::All,
                ParamGroupMatcher::Explicit(_) => {
                    let mut matchers = (*param_group_matchers).clone();
                    matchers.push(other);
                    Self::Combined(Arc::new(matchers))
                }
                ParamGroupMatcher::Path(_) => {
                    let mut matchers = (*param_group_matchers).clone();
                    matchers.push(other);
                    Self::Combined(Arc::new(matchers))
                }
                ParamGroupMatcher::Combined(other_matchers) => {
                    let mut matchers = (*param_group_matchers).clone();
                    matchers.append(&mut (*other_matchers).clone());
                    Self::Combined(Arc::new(matchers))
                }
            },
            _ => panic!(
                "`push_combined` should only be called on a ParamGroupMatcher::Combined variant."
            ),
        }
    }

    pub(crate) fn fuse(self, other: &Self) -> Self {
        match (self.clone(), other.clone()) {
            (ParamGroupMatcher::All, _) => Self::All,
            (_, ParamGroupMatcher::All) => Self::All,
            (ParamGroupMatcher::Explicit(_), ParamGroupMatcher::Combined(_)) => {
                other.clone().push_combined(self)
            }
            (ParamGroupMatcher::Path(_), ParamGroupMatcher::Combined(_)) => {
                other.clone().push_combined(self)
            }
            (ParamGroupMatcher::Combined(_), ParamGroupMatcher::Explicit(_)) => {
                self.push_combined(other.clone())
            }
            (ParamGroupMatcher::Combined(_), ParamGroupMatcher::Path(_)) => {
                self.push_combined(other.clone())
            }
            (ParamGroupMatcher::Combined(_), ParamGroupMatcher::Combined(_)) => {
                self.push_combined(other.clone())
            }
            _ => ParamGroupMatcher::Combined(Arc::new(vec![self, other.clone()])),
        }
    }
}

#[derive(Clone)]
enum PathMatcher {
    Exact(Vec<String>),
    Regex(Vec<Regex>),
    Include(Vec<String>),
}

impl PathMatcher {
    pub(crate) fn matches(&self, path: &str) -> bool {
        match self {
            PathMatcher::Exact(paths) => paths.iter().any(|p| p == path),
            PathMatcher::Regex(regexs) => regexs.iter().any(|r| r.is_match(path)),
            PathMatcher::Include(includes) => includes.iter().all(|inc| path.contains(inc)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::SimpleLinear;

    #[test]
    fn all_matches_any_parameter() {
        let group = ParamGroup::all();
        let id = ParamId::new();

        assert!(group.matches(&id, None).unwrap());
    }

    #[test]
    fn explicit_matches_only_selected_ids() {
        let id = ParamId::new();
        let other_id = ParamId::new();
        let group = ParamGroup::from_ids(vec![id.clone()]);

        assert!(group.matches(&id, None).unwrap());
        assert!(!group.matches(&other_id, None).unwrap());
    }

    #[test]
    fn path_matcher_requires_path_and_matches_exactly() {
        let group = ParamGroup::from_path("model.backbone.weight");
        let id = ParamId::new();

        assert!(group.matches(&id, Some("model.backbone.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.backbone.bias")).unwrap());

        let err = group.matches(&id, None).unwrap_err();
        assert!(matches!(err, ParamGroupError::InvalidParameter(_)));
    }

    #[test]
    fn predicate_matcher_matches_substrings() {
        let group = ParamGroup::from_predicate("backbone");
        let id = ParamId::new();

        assert!(group.matches(&id, Some("model.backbone.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.other.weight")).unwrap());
    }

    #[test]
    fn regex_matcher_matches_pattern() {
        let group = ParamGroup::from_regex(r"^model\.layer\.[0-9]+\.weight$").unwrap();
        let id = ParamId::new();

        assert!(group.matches(&id, Some("model.layer.3.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.layer.weight")).unwrap());
    }

    #[test]
    fn ids_from_module_collects_all_param_ids() {
        let device = crate::test_device();
        let module = SimpleLinear::new(4, 8, &device);
        let weight_id = module.weight.id;
        let bias_id = module.bias.as_ref().unwrap().id;
        let group = ParamGroup::ids_from_module(module);

        assert!(group.matches(&weight_id, Some("weight")).unwrap());
        assert!(group.matches(&bias_id, Some("bias")).unwrap());
    }

    #[test]
    fn fuse_combines_multiple_groups() {
        let id = ParamId::new();
        let group1 = ParamGroup::from_ids(vec![id.clone()]);
        let group2 = ParamGroup::from_path("model.layer.weight");
        let fused = group1.fuse(&group2);

        assert!(fused.matches(&id, Some("model.other.bias")).unwrap());
        assert!(
            fused
                .matches(&ParamId::new(), Some("model.layer.weight"))
                .unwrap()
        );
        assert!(
            !fused
                .matches(&ParamId::new(), Some("model.layer.bias"))
                .unwrap()
        );
    }

    #[test]
    fn exclude_removes_matching_ids_from_a_group() {
        let id = ParamId::new();
        let excluded_id = ParamId::new();
        let mut group = ParamGroup::from_ids(vec![id.clone(), excluded_id.clone()]);
        let exclude_group = ParamGroup::from_ids(vec![excluded_id.clone()]);

        group.exclude(&exclude_group);

        assert!(group.matches(&id, None).unwrap());
        assert!(!group.matches(&excluded_id, None).unwrap());
    }

    #[test]
    fn exclude_filters_path_matches() {
        let id = ParamId::new();
        let mut group = ParamGroup::from_predicate("backbone");
        group.exclude(&ParamGroup::from_path("model.backbone.bias"));

        assert!(group.matches(&id, Some("model.backbone.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.backbone.bias")).unwrap());
    }

    #[test]
    fn exclude_ignores_excludes_on_the_provided_group() {
        let id = ParamId::new();
        let mut excluded = ParamGroup::from_path("model.backbone.bias");
        excluded.exclude(&ParamGroup::from_path("model.backbone.weight"));

        let mut group = ParamGroup::from_path("model.backbone.weight");
        group.exclude(&excluded);

        assert!(group.matches(&id, Some("model.backbone.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.backbone.bias")).unwrap());
    }
}
