use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use ssh2::Session;

use crate::run_support::RunMsg;

/// The result of a streamed remote command.
pub(crate) enum ExecOutcome {
    /// The command finished with this exit status.
    Exited(i32),
    /// The cancel flag was set; the channel was closed and the run should stop.
    Canceled,
}

/// Rate-limits [`RunMsg::Progress`] updates so a long operation shows liveness without flooding.
pub(crate) struct Throttle<'a> {
    tx: &'a Sender<RunMsg>,
    last: Instant,
}

impl<'a> Throttle<'a> {
    pub fn new(tx: &'a Sender<RunMsg>) -> Self {
        Self {
            tx,
            last: Instant::now() - Duration::from_secs(1),
        }
    }

    /// Send `message` unless one was sent within the last 200ms.
    pub fn set(&mut self, message: String) {
        if self.last.elapsed() >= Duration::from_millis(200) {
            let _ = self.tx.send(RunMsg::Progress(message));
            self.last = Instant::now();
        }
    }
}

/// Which family of remote OS we're talking to. Linux and macOS share POSIX semantics; Windows
/// (OpenSSH default shell `cmd.exe`) needs its own command forms.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Platform {
    Unix,
    Windows,
}

impl Platform {
    pub fn label(self) -> &'static str {
        match self {
            Platform::Unix => "unix",
            Platform::Windows => "windows",
        }
    }
}

/// An authenticated SSH connection with an open SFTP channel, plus the remote's platform and the
/// temp/home directories resolved once at connect time.
pub(crate) struct Remote {
    session: Session,
    sftp: ssh2::Sftp,
    pub platform: Platform,
    /// Remote temp directory with forward slashes (`/tmp`, or `%TEMP%` on Windows).
    pub temp_dir: String,
    /// Remote home directory with forward slashes.
    pub home: String,
}

impl Remote {
    /// Connect and authenticate. `host_spec` is either `user@host[:port]` or a bare `~/.ssh/config`
    /// alias (e.g. `tower`), whose `HostName`/`User`/`Port`/`IdentityFile` are resolved. An explicit
    /// `user@`/`:port` overrides the config. A non-empty `password` uses password auth; otherwise
    /// the ssh-agent, the config's `IdentityFile`, then the usual `~/.ssh` key files are tried.
    pub fn connect(host_spec: &str, password: &str) -> Result<Self, String> {
        let target = resolve_target(host_spec.trim())?;
        let socket = target
            .addr
            .to_socket_addrs()
            .map_err(|e| format!("resolve {}: {e}", target.addr))?
            .next()
            .ok_or_else(|| format!("no address for {}", target.addr))?;

        let tcp = TcpStream::connect_timeout(&socket, Duration::from_secs(10))
            .map_err(|e| format!("connect {}: {e}", target.addr))?;
        let mut session = Session::new().map_err(|e| e.to_string())?;
        session.set_tcp_stream(tcp);
        session.handshake().map_err(|e| format!("handshake: {e}"))?;

        if password.is_empty() {
            authenticate_with_keys(&session, &target.user, target.identity.as_deref())?;
        } else {
            session
                .userauth_password(&target.user, password)
                .map_err(|e| format!("password auth: {e}"))?;
        }
        if !session.authenticated() {
            return Err("authentication failed".to_string());
        }

        let sftp = session.sftp().map_err(|e| format!("open sftp: {e}"))?;
        let platform = detect_platform(&session);
        let (temp_dir, home) = resolve_dirs(&session, platform);
        Ok(Self {
            session,
            sftp,
            platform,
            temp_dir,
            home,
        })
    }

    /// Run `command` and stream merged stdout+stderr as lines, returning the exit status.
    pub fn exec_stream(
        &self,
        command: &str,
        cancel: &AtomicBool,
        tx: &Sender<RunMsg>,
    ) -> Result<ExecOutcome, String> {
        let mut channel = self.session.channel_session().map_err(|e| e.to_string())?;
        channel
            .handle_extended_data(ssh2::ExtendedData::Merge)
            .map_err(|e| e.to_string())?;
        channel.exec(command).map_err(|e| e.to_string())?;

        // Non-blocking reads so the cancel flag is polled between chunks even while the remote is
        // quiet (e.g. compiling with no output).
        self.session.set_blocking(false);
        let canceled = self.pump(&mut channel, cancel, tx);
        self.session.set_blocking(true);

        if canceled {
            let _ = channel.close();
            return Ok(ExecOutcome::Canceled);
        }
        channel.wait_close().ok();
        Ok(ExecOutcome::Exited(channel.exit_status().unwrap_or(-1)))
    }

    /// Drain the channel, emitting complete lines, until EOF or the cancel flag is set. Returns
    /// `true` if it stopped because of cancellation.
    fn pump(&self, channel: &mut ssh2::Channel, cancel: &AtomicBool, tx: &Sender<RunMsg>) -> bool {
        let mut pending = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            if cancel.load(Ordering::Relaxed) {
                flush_line(&mut pending, tx);
                return true;
            }
            match channel.read(&mut chunk) {
                Ok(0) => {
                    flush_line(&mut pending, tx);
                    return false;
                }
                Ok(n) => emit_lines(&mut pending, &chunk[..n], tx),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(40));
                }
                Err(_) => {
                    flush_line(&mut pending, tx);
                    return false;
                }
            }
        }
    }

    /// Best-effort kill of the remote run whose command line contains `pattern` (the run id),
    /// used when a remote run is canceled mid-flight. No-op on Windows.
    pub fn kill_matching(&self, pattern: &str) {
        if self.platform != Platform::Unix {
            return;
        }
        if let Ok(mut channel) = self.session.channel_session() {
            let _ = channel.exec(&format!("pkill -f \"{pattern}\""));
            let mut sink = String::new();
            let _ = channel.read_to_string(&mut sink);
            let _ = channel.wait_close();
        }
    }

    /// Wrap the OS-independent `tail` (the `cargo run …` command) with the platform's directory
    /// change and colour-forcing env var.
    pub fn run_command(&self, workdir: &str, tail: &str) -> String {
        match self.platform {
            Platform::Unix => format!("cd \"{workdir}\" && CARGO_TERM_COLOR=always {tail}"),
            Platform::Windows => {
                format!("cd /d \"{workdir}\" && set CARGO_TERM_COLOR=always&& {tail}")
            }
        }
    }

    /// Recursively create `dir` over SFTP (portable across OSes), ignoring already-existing parts.
    pub fn ensure_dir(&self, dir: &str) -> Result<(), String> {
        let absolute = dir.starts_with('/');
        let mut acc = String::new();
        for part in dir.split('/').filter(|p| !p.is_empty()) {
            if acc.is_empty() {
                if absolute {
                    acc.push('/');
                }
            } else {
                acc.push('/');
            }
            acc.push_str(part);
            let _ = self.sftp.mkdir(Path::new(&acc), 0o755);
        }
        Ok(())
    }

    /// Listing of `dir` as `relative path -> (size, mtime secs)`, gathered by walking the remote
    /// tree over SFTP. Portable across OSes and avoids depending on a specific `find`. Emits scan
    /// progress so a large existing tree doesn't look stalled.
    pub fn manifest(&self, dir: &str, tx: &Sender<RunMsg>) -> HashMap<String, (u64, u64)> {
        let mut map = HashMap::new();
        let mut throttle = Throttle::new(tx);
        self.walk(dir, dir, &mut map, &mut throttle);
        map
    }

    fn walk(
        &self,
        root: &str,
        dir: &str,
        out: &mut HashMap<String, (u64, u64)>,
        throttle: &mut Throttle,
    ) {
        let Ok(entries) = self.sftp.readdir(Path::new(dir)) else {
            return;
        };
        for (path, stat) in entries {
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.is_empty() || name == "." || name == ".." {
                continue;
            }
            let full = format!("{dir}/{name}");
            if stat.is_dir() {
                if !super::sync::is_skipped(&name) {
                    self.walk(root, &full, out, throttle);
                }
            } else if stat.is_file() {
                let rel = full.strip_prefix(root).unwrap_or(&full).trim_start_matches('/');
                out.insert(
                    rel.to_string(),
                    (stat.size.unwrap_or(0), stat.mtime.unwrap_or(0)),
                );
                throttle.set(format!("scanning remote… {} files", out.len()));
            }
        }
    }

    /// Upload `local` to `remote`, setting the remote mtime so later syncs can skip it.
    pub fn upload(&self, local: &Path, remote: &str, mtime: u64) -> Result<(), String> {
        let data = std::fs::read(local).map_err(|e| e.to_string())?;
        let mut file = self
            .sftp
            .create(Path::new(remote))
            .map_err(|e| format!("create {remote}: {e}"))?;
        file.write_all(&data).map_err(|e| e.to_string())?;
        drop(file);
        let stat = ssh2::FileStat {
            size: None,
            uid: None,
            gid: None,
            perm: None,
            atime: Some(mtime),
            mtime: Some(mtime),
        };
        let _ = self.sftp.setstat(Path::new(remote), stat);
        Ok(())
    }

    /// Read the sync-stamp token at `remote_root/.sync-stamp`, or `None` if it's absent — a cheap
    /// one-round-trip check used to validate the local sync cache.
    pub fn read_stamp(&self, remote_root: &str) -> Option<String> {
        let mut file = self
            .sftp
            .open(Path::new(&format!("{remote_root}/.sync-stamp")))
            .ok()?;
        let mut token = String::new();
        file.read_to_string(&mut token).ok()?;
        Some(token.trim().to_string())
    }

    /// Write the sync-stamp token marking `remote_root` as synced for this snapshot.
    pub fn write_stamp(&self, remote_root: &str, token: &str) -> Result<(), String> {
        let mut file = self
            .sftp
            .create(Path::new(&format!("{remote_root}/.sync-stamp")))
            .map_err(|e| e.to_string())?;
        file.write_all(token.as_bytes()).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Download `remote` into `local`, creating parent directories.
    pub fn download(&self, remote: &str, local: &Path) -> Result<(), String> {
        let mut file = self
            .sftp
            .open(Path::new(remote))
            .map_err(|e| format!("open {remote}: {e}"))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| e.to_string())?;
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(local, data).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Probe the remote OS: `uname -s` yields `Linux`/`Darwin` on POSIX hosts and fails on Windows
/// `cmd.exe`, so an absent/unexpected result means Windows.
fn detect_platform(session: &Session) -> Platform {
    let out = capture_on(session, "uname -s").to_lowercase();
    if out.contains("linux") || out.contains("darwin") || out.contains("bsd") {
        Platform::Unix
    } else {
        Platform::Windows
    }
}

/// Resolve the remote temp and home directories for the detected platform, normalising to `/`.
fn resolve_dirs(session: &Session, platform: Platform) -> (String, String) {
    match platform {
        Platform::Unix => {
            let home = capture_on(session, "printf %s \"$HOME\"").trim().to_string();
            ("/tmp".to_string(), home)
        }
        Platform::Windows => {
            // A non-`cmd.exe` default shell may not expand `%VAR%`; fall back when that happens.
            let temp = win_var(session, "%TEMP%", "C:/Windows/Temp");
            let home = win_var(session, "%USERPROFILE%", "C:/Users/Public");
            (temp, home)
        }
    }
}

/// Read a Windows env var via `cmd.exe`, returning `fallback` if the shell didn't expand it.
fn win_var(session: &Session, var: &str, fallback: &str) -> String {
    let value = normalize(&capture_on(session, &format!("echo {var}")));
    if value.is_empty() || value.contains('%') {
        fallback.to_string()
    } else {
        value
    }
}

fn capture_on(session: &Session, command: &str) -> String {
    let Ok(mut channel) = session.channel_session() else {
        return String::new();
    };
    if channel.exec(command).is_err() {
        return String::new();
    }
    let mut out = String::new();
    let _ = channel.read_to_string(&mut out);
    channel.wait_close().ok();
    out
}

fn normalize(path: &str) -> String {
    path.trim().replace('\\', "/")
}

/// Append `data` to `pending` and emit every complete (newline-terminated) line.
fn emit_lines(pending: &mut Vec<u8>, data: &[u8], tx: &Sender<RunMsg>) {
    pending.extend_from_slice(data);
    while let Some(newline) = pending.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = pending.drain(..=newline).collect();
        let text = String::from_utf8_lossy(&line);
        let _ = tx.send(RunMsg::Line(text.trim_end_matches(['\n', '\r']).to_string()));
    }
}

/// Emit any buffered trailing bytes as a final line.
fn flush_line(pending: &mut Vec<u8>, tx: &Sender<RunMsg>) {
    if !pending.is_empty() {
        let text = String::from_utf8_lossy(pending);
        let _ = tx.send(RunMsg::Line(text.trim_end_matches(['\n', '\r']).to_string()));
        pending.clear();
    }
}

/// A connection target resolved from the host field and `~/.ssh/config`.
struct Target {
    addr: String,
    user: String,
    identity: Option<PathBuf>,
}

/// Turn `user@host:port` or a bare `~/.ssh/config` alias into a concrete address/user/identity.
/// Explicit `user@` and `:port` win over the config; the config fills whatever is left.
fn resolve_target(input: &str) -> Result<Target, String> {
    let (explicit_user, rest) = match input.split_once('@') {
        Some((user, rest)) => (Some(user.to_string()), rest.to_string()),
        None => (None, input.to_string()),
    };
    let (token, explicit_port) = match rest.rsplit_once(':') {
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) => {
            (host.to_string(), port.parse::<u16>().ok())
        }
        _ => (rest, None),
    };
    if token.is_empty() {
        return Err("empty host".to_string());
    }

    let cfg = super::ssh_config::lookup(&token);
    let host = cfg.hostname.unwrap_or(token);
    let port = explicit_port.or(cfg.port).unwrap_or(22);
    let user = explicit_user
        .or(cfg.user)
        .or_else(|| std::env::var("USER").ok())
        .ok_or("no user (use user@host or set User in ~/.ssh/config)")?;
    Ok(Target {
        addr: format!("{host}:{port}"),
        user,
        identity: cfg.identity_file,
    })
}

fn authenticate_with_keys(
    session: &Session,
    user: &str,
    identity: Option<&Path>,
) -> Result<(), String> {
    if session.userauth_agent(user).is_ok() && session.authenticated() {
        return Ok(());
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let mut keys: Vec<PathBuf> = identity.map(Path::to_path_buf).into_iter().collect();
    for name in ["id_ed25519", "id_rsa", "id_ecdsa"] {
        keys.push(Path::new(&home).join(".ssh").join(name));
    }
    for key in keys {
        if key.exists()
            && session
                .userauth_pubkey_file(user, None, &key, None)
                .is_ok()
            && session.authenticated()
        {
            return Ok(());
        }
    }
    Err("no usable ssh-agent identity or key (set a password instead)".to_string())
}
