use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use ignore::WalkBuilder;

use super::session::{Remote, Throttle};
use crate::example_dir;
use crate::run_support::RunMsg;

/// Directories pruned when scanning the *remote* tree — a performance guard so a post-build
/// `target/` (which the remote cargo run creates) doesn't stall the SFTP walk. Local file
/// selection uses gitignore instead (see [`collect_files`]).
const REMOTE_SKIP_DIRS: [&str; 3] = ["target", ".git", "runs"];

/// Whether a directory name is pruned from the remote manifest walk.
pub(crate) fn is_skipped(name: &str) -> bool {
    REMOTE_SKIP_DIRS.contains(&name)
}

type Manifest = HashMap<String, (u64, u64)>;
type FileList = Vec<(String, u64, u64)>;

/// Push `local_root` to `remote_root`, uploading only new/changed files (by size or mtime) and
/// streaming progress.
///
/// To avoid re-walking the whole remote tree every run (slow over SFTP), the last synced state is
/// cached locally and tagged with a stamp also written on the remote. When the remote stamp still
/// matches the cache, the remote scan is skipped and only local changes are considered. A missing
/// or mismatched stamp (first sync, or the remote's temp dir wiped on reboot) falls back to a full
/// scan. `force` bypasses the cache entirely.
pub(crate) fn sync_tree(
    remote: &Remote,
    local_root: &Path,
    remote_root: &str,
    host: &str,
    force: bool,
    cancel: &AtomicBool,
    tx: &Sender<RunMsg>,
) -> Result<bool, String> {
    let cache_file = cache_path(host, remote_root);
    let cached = if force { None } else { load_cache(&cache_file) };
    let remote_stamp = remote.read_stamp(remote_root);
    let trust = matches!((&cached, &remote_stamp), (Some((s, _)), Some(rs)) if s == rs);

    let baseline: Manifest = if trust {
        let _ = tx.send(RunMsg::Progress("checking local changes…".to_string()));
        cached.as_ref().unwrap().1.clone()
    } else {
        let _ = tx.send(RunMsg::Progress(format!("scanning remote {remote_root}…")));
        remote.manifest(remote_root, tx)
    };

    let mut files = FileList::new();
    collect_files(local_root, &mut files)?;

    let pending: Vec<&(String, u64, u64)> = files
        .iter()
        .filter(|(rel, size, mtime)| match baseline.get(rel) {
            Some((base_size, base_mtime)) => base_size != size || base_mtime < mtime,
            None => true,
        })
        .collect();

    if pending.is_empty() {
        let _ = tx.send(RunMsg::Line(format!(
            "  {remote_root}: up to date ({} files)",
            files.len()
        )));
        // Establish a stamp + cache so the next run can take the fast path.
        let stamp = match &cached {
            Some((stamp, _)) if trust => stamp.clone(),
            _ => {
                let stamp = new_stamp();
                remote.write_stamp(remote_root, &stamp)?;
                stamp
            }
        };
        save_cache(&cache_file, &stamp, &files);
        return Ok(true);
    }

    let total = pending.len();
    let total_bytes: u64 = pending.iter().map(|(_, size, _)| *size).sum();
    let _ = tx.send(RunMsg::Line(format!(
        "  {remote_root}: uploading {total} files ({})…",
        human_bytes(total_bytes)
    )));

    let mut created: HashSet<String> = HashSet::new();
    let mut done_bytes = 0u64;
    let mut throttle = Throttle::new(tx);
    for (index, (rel, size, mtime)) in pending.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            // Leave the stamp/cache unwritten so the next run re-checks and resumes.
            return Ok(false);
        }
        let remote_path = format!("{remote_root}/{rel}");
        if let Some(parent) = Path::new(&remote_path).parent() {
            let parent = parent.to_string_lossy().into_owned();
            if created.insert(parent.clone()) {
                remote.ensure_dir(&parent)?;
            }
        }
        remote.upload(&local_root.join(rel), &remote_path, *mtime)?;
        done_bytes += *size;
        throttle.set(format!(
            "uploading to {remote_root}: {}/{total} files, {}/{}",
            index + 1,
            human_bytes(done_bytes),
            human_bytes(total_bytes)
        ));
    }

    // Record the synced state so the next run can skip the remote scan. Reuse the stamp on the fast
    // path (it's already on the remote); mint and write a fresh one after a full scan.
    let stamp = match &cached {
        Some((stamp, _)) if trust => stamp.clone(),
        _ => {
            let stamp = new_stamp();
            remote.write_stamp(remote_root, &stamp)?;
            stamp
        }
    };
    save_cache(&cache_file, &stamp, &files);

    let _ = tx.send(RunMsg::Line(format!(
        "  {remote_root}: uploaded {total} files ({}), {} unchanged",
        human_bytes(total_bytes),
        files.len() - total
    )));
    Ok(true)
}

/// Collect regular files under `root` as `(path relative to root, size, mtime secs)`, honouring
/// `.gitignore` (and `.ignore`/exclude) files. `.git` is always skipped; dotfiles are kept (so
/// e.g. `.cargo/config.toml` still syncs). Symlinks are not followed.
fn collect_files(root: &Path, out: &mut FileList) -> Result<(), String> {
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .follow_links(false)
        .filter_entry(|entry| entry.file_name() != ".git")
        .build();
    for entry in walker {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let Ok(rel) = entry.path().strip_prefix(root) else {
            continue;
        };
        out.push((rel.to_string_lossy().replace('\\', "/"), meta.len(), mtime));
    }
    Ok(())
}

fn new_stamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .to_string()
}

fn cache_path(host: &str, remote_root: &str) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    host.hash(&mut hasher);
    remote_root.hash(&mut hasher);
    example_dir()
        .join(".sync-cache")
        .join(format!("{:016x}.tsv", hasher.finish()))
}

/// Load a cached snapshot: `(stamp, path -> (size, mtime))`. The first line is `stamp\t<value>`.
fn load_cache(path: &Path) -> Option<(String, Manifest)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let stamp = lines.next()?.strip_prefix("stamp\t")?.to_string();
    let mut manifest = Manifest::new();
    for line in lines {
        let mut parts = line.split('\t');
        if let (Some(rel), Some(size), Some(mtime)) = (parts.next(), parts.next(), parts.next())
            && let (Ok(size), Ok(mtime)) = (size.parse(), mtime.parse())
        {
            manifest.insert(rel.to_string(), (size, mtime));
        }
    }
    Some((stamp, manifest))
}

fn save_cache(path: &Path, stamp: &str, files: &FileList) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut out = format!("stamp\t{stamp}\n");
    for (rel, size, mtime) in files {
        out.push_str(&format!("{rel}\t{size}\t{mtime}\n"));
    }
    let _ = std::fs::write(path, out);
}

/// Format a byte count as a short human-readable string (e.g. `12.3 MB`).
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::collect_files;
    use std::fs;

    fn write(path: &std::path::Path, contents: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn collect_files_respects_gitignore() {
        let root = std::env::temp_dir().join(format!("burn-sync-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write(&root.join(".gitignore"), "ignored.txt\nbuild/\n");
        write(&root.join("keep.txt"), "a");
        write(&root.join("ignored.txt"), "b");
        write(&root.join("build/artifact.o"), "c");
        write(&root.join(".cargo/config.toml"), "d");
        write(&root.join(".git/HEAD"), "e");

        let mut files = Vec::new();
        collect_files(&root, &mut files).unwrap();
        let rels: Vec<&String> = files.iter().map(|(rel, _, _)| rel).collect();

        assert!(rels.iter().any(|r| *r == "keep.txt"));
        assert!(rels.iter().any(|r| *r == ".cargo/config.toml"), "dotfiles kept");
        assert!(!rels.iter().any(|r| *r == "ignored.txt"), "gitignore honoured");
        assert!(!rels.iter().any(|r| *r == "build/artifact.o"), "ignored dir pruned");
        assert!(!rels.iter().any(|r| r.starts_with(".git/")), ".git skipped");

        let _ = fs::remove_dir_all(&root);
    }
}
