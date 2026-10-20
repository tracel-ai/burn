//! Components fed from more than one call site.

use std::fmt::{self, Display};

use burn_instrument::Component;

use crate::Report;

/// A note per component fed from two places, or nothing where none was.
///
/// After the table, because it is a fact about a component a reader has already seen, and
/// never instead of one: the tree is unchanged and the merge stands.
pub(super) struct MergeNotes<'r, 'a> {
    report: &'r Report<'a>,
}

impl<'r, 'a> MergeNotes<'r, 'a> {
    pub(super) fn new(report: &'r Report<'a>) -> Self {
        Self { report }
    }
}

impl Display for MergeNotes<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (under, component) in merged(&self.report.measurement().breakdown.tree) {
            let sites: Vec<String> = component.sites.iter().map(|it| site(it)).collect();
            let under = match under.is_empty() {
                true => String::new(),
                false => format!(" under `{under}`"),
            };
            writeln!(
                f,
                "\n  note: `{}`{under} was entered from {} sites ({})",
                component.label,
                sites.len(),
                sites.join(", ")
            )?;
        }
        Ok(())
    }
}

/// Components fed from more than one call site, each with the path that names it.
///
/// A merge is not a fault — one component entered 36 times is what makes a stack
/// readable, and two sites under one label is the same mechanism. It is worth a
/// note only because it is also how one component comes to hold two different jobs:
/// two entry points that both write a cache under one label would sum two
/// different writes under one heading.
///
/// A site is a function, so this reports two *entry points*, never two roles of
/// one entry point — that is what a caller-supplied label is for.
fn merged(tree: &[Component]) -> Vec<(String, &Component)> {
    fn descend<'a>(component: &'a Component, under: &str, out: &mut Vec<(String, &'a Component)>) {
        if component.sites.len() > 1 {
            out.push((under.to_string(), component));
        }
        for child in &component.children {
            descend(child, &component.label, out);
        }
    }
    let mut out = Vec::new();
    for root in tree {
        descend(root, "", &mut out);
    }
    out
}

/// A site under the shortest path that still says where to look.
///
/// The crate is dropped, since a model's components mostly come from one or two, and
/// so is a `base` module, which holds a folder's one implementation — so
/// `my_model::rope::base::apply` reads as `rope::apply`, which is what a
/// reader greps for.
fn site(site: &str) -> String {
    site.split("::")
        .skip(1)
        .filter(|segment| *segment != "base")
        .collect::<Vec<_>>()
        .join("::")
}
