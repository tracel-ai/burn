use std::path::{Path, PathBuf};

use crate::example_dir;

/// A local repository that must exist on the remote for the build to succeed. `name` is the
/// directory it is placed under inside the remote base dir.
pub(crate) struct SyncRepo {
    pub name: String,
    pub local: PathBuf,
}

/// Walk up from the example dir to the workspace root — the nearest ancestor whose `Cargo.toml`
/// declares a `[workspace]`. Falls back to the example dir itself when the example is standalone
/// (e.g. after it moves to its own repo), so the rest of the flow keeps working.
pub(crate) fn workspace_root() -> PathBuf {
    let mut dir = example_dir();
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("Cargo.toml"))
            && text
                .lines()
                .any(|line| line.trim_start().starts_with("[workspace]"))
        {
            return dir;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => return example_dir(),
        }
    }
}

/// Name of the workspace-root directory, used as its folder under the remote base dir.
pub(crate) fn root_name() -> String {
    dir_name(&workspace_root())
}

/// The example dir relative to the workspace root (e.g. `examples/autotune-observability`, or
/// empty when the example *is* the root).
pub(crate) fn example_rel() -> PathBuf {
    example_dir()
        .strip_prefix(workspace_root())
        .map(Path::to_path_buf)
        .unwrap_or_default()
}

/// Repos to push to the remote: always the workspace root, plus any sibling repo referenced by a
/// local `path = "../…"` dependency (cubecl / cubek) so a locally-patched build reproduces remotely.
pub(crate) fn repos_to_sync() -> Vec<SyncRepo> {
    let root = workspace_root();
    let mut repos = vec![SyncRepo {
        name: dir_name(&root),
        local: root.clone(),
    }];
    for local in external_repos(&root) {
        repos.push(SyncRepo {
            name: dir_name(&local),
            local,
        });
    }
    repos
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "repo".to_string())
}

/// Local sibling repos the build depends on, resolved to absolute (deduplicated) paths. Returns
/// nothing when there is no workspace manifest or no such dependency, in which case only the root
/// is synced.
fn external_repos(root: &Path) -> Vec<PathBuf> {
    let Ok(text) = std::fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };

    let mut repos: Vec<PathBuf> = Vec::new();
    for repo_rel in external_repo_rels(&text) {
        if let Ok(repo) = root.join(&repo_rel).canonicalize()
            && !repos.contains(&repo)
        {
            repos.push(repo);
        }
    }
    repos
}

/// Parse a workspace `Cargo.toml` for active (uncommented) dependency-table path overrides that
/// point at a *sibling* repo (`path = "../…"`), and return each such repo's root relative path
/// (deduplicated). This covers both `[patch."…"]` sections and plain `path` dependencies declared
/// under `[dependencies]` / `[workspace.dependencies]` (how burn overrides cubecl / cubek locally).
/// Intra-workspace path deps (e.g. `crates/burn-cubecl`) are ignored — the root already covers them.
fn external_repo_rels(manifest: &str) -> Vec<String> {
    let mut rels: Vec<String> = Vec::new();
    let mut in_deps = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_deps = line.starts_with("[patch.") || line.contains("dependencies]");
            continue;
        }
        if !in_deps {
            continue;
        }
        let Some(path) = patch_path(line) else { continue };
        // Only sibling repos (`../…`) need syncing; intra-workspace paths stay with the root.
        if !path.starts_with("..") {
            continue;
        }
        // `path` is like `../cubecl/crates/cubecl`; the repo root is the part before `/crates/`.
        let repo_rel = path.split("/crates/").next().unwrap_or(&path).to_string();
        if !rels.contains(&repo_rel) {
            rels.push(repo_rel);
        }
    }
    rels
}

/// Extract the `path = "…"` value from a patch dependency line, if present.
fn patch_path(line: &str) -> Option<String> {
    let after_key = &line[line.find("path")?..];
    let open = after_key.find('"')? + 1;
    let value = &after_key[open..];
    let close = value.find('"')?;
    Some(value[..close].to_string())
}

#[cfg(test)]
mod tests {
    use super::{external_repo_rels, patch_path};

    #[test]
    fn parses_patch_path() {
        assert_eq!(
            patch_path(r#"cubecl = { path = "../cubecl/crates/cubecl" }"#).as_deref(),
            Some("../cubecl/crates/cubecl")
        );
        assert_eq!(patch_path("cubecl = { version = \"0.10.0\" }"), None);
    }

    #[test]
    fn repo_root_is_part_before_crates() {
        let path = "../cubecl/crates/cubecl";
        assert_eq!(path.split("/crates/").next(), Some("../cubecl"));
    }

    #[test]
    fn detects_sibling_repos_from_workspace_dependencies() {
        // Mirrors burn's layout: cubecl/cubek overridden via `[workspace.dependencies]` (no
        // `[patch]` section), alongside intra-workspace crate deps and a commented-out git source.
        let manifest = r#"
[workspace.dependencies]
burn-cubecl = { path = "crates/burn-cubecl", version = "0.22.0-pre.1" }
# cubecl = { git = "https://github.com/tracel-ai/cubecl", rev = "abc" }
cubecl = { path = "../cubecl/crates/cubecl", default-features = false }
cubecl-common = { path = "../cubecl/crates/cubecl-common", default-features = false }
cubek = { path = "../cubek/crates/cubek", default-features = false }

[profile.dev]
opt-level = 0
"#;
        assert_eq!(external_repo_rels(manifest), vec!["../cubecl", "../cubek"]);
    }

    #[test]
    fn detects_sibling_repos_from_patch_section() {
        let manifest = r#"
[patch.crates-io]
cubecl = { path = "../cubecl/crates/cubecl" }
"#;
        assert_eq!(external_repo_rels(manifest), vec!["../cubecl"]);
    }

    #[test]
    fn ignores_intra_workspace_and_registry_deps() {
        let manifest = r#"
[workspace.dependencies]
burn-cubecl = { path = "crates/burn-cubecl" }
serde = { version = "1" }
"#;
        assert!(external_repo_rels(manifest).is_empty());
    }
}
