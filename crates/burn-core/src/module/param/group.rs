use crate::module::{Module, ModuleVisitor, Param};

use alloc::string::String;
#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;
#[cfg(feature = "std")]
use regex::Regex;

use burn_std::id::ParamId;
use burn_tensor::{Bool, Int, Tensor};

/// Errors tied to [ParamGroup]'s.
#[derive(Debug)]
pub enum ParamGroupError {
    /// Parameter used to match are invalid.
    InvalidParameter(String),
    /// Failed to create the group.
    GroupInitError(String),
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
#[derive(Clone)]
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

    /// Matches parameters that include the predicate in their paths (e.g., "backbone")
    pub fn from_predicate(path: impl Into<String>) -> Self {
        ParamGroup::from_predicates(vec![path])
    }

    /// Matches parameters that include all the predicates in their path (AND logic).
    /// (e.g., parameter path contains "backbone" and "linear")
    pub fn from_predicates(paths: Vec<impl Into<String>>) -> Self {
        Self {
            matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Include(
                paths.into_iter().map(|p| p.into()).collect(),
            ))),
            excludes: None,
        }
    }

    /// Matches parameters that include any of the predicates in their path (OR logic).
    /// (e.g., parameter path contains "backbone" or "linear")
    pub fn from_any_predicates(paths: Vec<impl Into<String>>) -> Self {
        let mut matchers: Vec<ParamGroupMatcher> = paths
            .into_iter()
            .map(|p| ParamGroupMatcher::Path(Arc::new(PathMatcher::Include(vec![p.into()]))))
            .collect();
        let mut main_matcher = if let Some(value) = matchers.pop() {
            value
        } else {
            return Self {
                matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Include(vec![]))),
                excludes: None,
            };
        };

        matchers
            .iter()
            .for_each(|m| main_matcher = main_matcher.clone().fuse(m));

        Self {
            matcher: main_matcher,
            excludes: None,
        }
    }

    #[cfg(feature = "std")]
    /// Matches parameters by regex pattern (e.g., "^model\.layer\.\d+$")
    ///
    /// # Errors
    /// Returns a [ParamGroupError::GroupInitError] if the string cannot be compiled into a valid regex.
    pub fn from_regex<S: AsRef<str>>(pattern: S) -> Result<Self, ParamGroupError> {
        ParamGroup::from_regexes(vec![pattern])
    }

    #[cfg(feature = "std")]
    /// Matches parameters for all the regex patterns (AND logic).
    /// (e.g., "^encoder\.layer\.\d+", and "bias$" )
    ///
    /// # Errors
    /// Returns a [ParamGroupError::GroupInitError] if the strings cannot be compiled into a valid regex.
    pub fn from_regexes<I, S>(patterns: I) -> Result<Self, ParamGroupError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut new_patterns = vec![];
        for pattern in patterns {
            new_patterns.push(
                Regex::new(pattern.as_ref())
                    .map_err(|e| ParamGroupError::GroupInitError(e.to_string()))?,
            );
        }
        Ok(Self {
            matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Regex(new_patterns))),
            excludes: None,
        })
    }

    #[cfg(feature = "std")]
    /// Matches parameters for any the regex patterns (OR logic).
    /// (e.g., "^encoder\.layer\.\d+$", or "^decoder\.layer\.\d+$" )
    ///
    /// # Errors
    /// Returns a [ParamGroupError::GroupInitError] if the strings cannot be compiled into a valid regex.
    pub fn from_any_regexes<I, S>(patterns: I) -> Result<Self, ParamGroupError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut matchers = vec![];
        for pattern in patterns {
            matchers.push(ParamGroupMatcher::Path(Arc::new(PathMatcher::Regex(vec![
                Regex::new(pattern.as_ref())
                    .map_err(|e| ParamGroupError::GroupInitError(e.to_string()))?,
            ]))));
        }

        let mut main_matcher = if let Some(value) = matchers.pop() {
            value
        } else {
            return Ok(Self {
                matcher: ParamGroupMatcher::Path(Arc::new(PathMatcher::Include(vec![]))),
                excludes: None,
            });
        };

        matchers
            .iter()
            .for_each(|m| main_matcher = main_matcher.clone().fuse(m));

        Ok(Self {
            matcher: main_matcher,
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

    /// Exclude the given group from the current group
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
            Self::Explicit(ids) => Ok(ids.contains(id)),
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
    #[cfg(feature = "std")]
    Regex(Vec<Regex>),
    Include(Vec<String>),
}

impl PathMatcher {
    pub(crate) fn matches(&self, path: &str) -> bool {
        match self {
            PathMatcher::Exact(paths) => paths.iter().any(|p| p == path),
            #[cfg(feature = "std")]
            PathMatcher::Regex(regexs) => regexs.iter().all(|r| r.is_match(path)),
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

    #[cfg(feature = "std")]
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

    #[test]
    fn from_any_predicates_matches_any_predicate() {
        let group = ParamGroup::from_any_predicates(vec!["backbone", "encoder"]);
        let id = ParamId::new();

        assert!(group.matches(&id, Some("model.backbone.weight")).unwrap());
        assert!(group.matches(&id, Some("model.encoder.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.decoder.weight")).unwrap());
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_any_regexes_matches_any_pattern() {
        let group =
            ParamGroup::from_any_regexes(vec![r"^model\.layer\.[0-9]+\.weight$", r"^model\.bias$"])
                .unwrap();
        let id = ParamId::new();

        assert!(group.matches(&id, Some("model.layer.3.weight")).unwrap());
        assert!(group.matches(&id, Some("model.bias")).unwrap());
        assert!(!group.matches(&id, Some("model.layer.weight")).unwrap());
        assert!(!group.matches(&id, Some("model.other.weight")).unwrap());
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_regexes_with_invalid_pattern() {
        let result = ParamGroup::from_regex(r"[invalid(");
        assert!(result.is_err());

        let result = ParamGroup::from_regexes(vec![r"[invalid("]);
        assert!(result.is_err());

        let result = ParamGroup::from_any_regexes(vec![r"[invalid("]);
        assert!(result.is_err());
    }
}
