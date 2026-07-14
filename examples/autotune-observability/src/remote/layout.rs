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

/// Repos to push to the remote: always the workspace root, plus any repo pointed to by an active
/// `[patch]` (cubecl / cubek) so a locally-patched build reproduces remotely.
pub(crate) fn repos_to_sync() -> Vec<SyncRepo> {
    let root = workspace_root();
    let mut repos = vec![SyncRepo {
        name: dir_name(&root),
        local: root.clone(),
    }];
    for local in active_patch_repos(&root) {
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

/// Parse the workspace `Cargo.toml` for active (uncommented) `[patch."…"]` sections and return
/// the local repo root each patched crate path points into (deduplicated). Returns nothing when
/// there is no workspace manifest or no active patch, in which case only the root is synced.
fn active_patch_repos(root: &Path) -> Vec<PathBuf> {
    let Ok(text) = std::fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };

    let mut repos: Vec<PathBuf> = Vec::new();
    let mut in_patch = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_patch = line.starts_with("[patch.");
            continue;
        }
        if !in_patch {
            continue;
        }
        if let Some(path) = patch_path(line) {
            // `path` is like `../cubecl/crates/cubecl`; the repo root is the part before `/crates/`.
            let repo_rel = path.split("/crates/").next().unwrap_or(&path);
            if let Ok(repo) = root.join(repo_rel).canonicalize()
                && !repos.contains(&repo)
            {
                repos.push(repo);
            }
        }
    }
    repos
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
    use super::patch_path;

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
}
