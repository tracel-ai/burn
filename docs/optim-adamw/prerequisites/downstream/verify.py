#!/usr/bin/env python3
"""Verify PR3 through local git dependencies, without Cargo source overrides (Python 3.9+)."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CUBECL_REV = "d88b0febb50e7b83a8548125f6a701c12f11a2ba"
CUBEK_REV = "3d3ab9680931c5903286bb30eb2b8ff156e1618b"
PATCHES = [
    ROOT / "docs/optim-adamw/prerequisites/cubecl-execution-stream.patch",
    ROOT / "docs/optim-adamw/experiments/pr3-local/foreach.patch",
    ROOT / "docs/optim-adamw/experiments/pr3-local/refinement/candidate-refinement.patch",
]

# The consumer has just one dependency: the public Burn facade, fetched by git.
MAIN_RS = r'''use burn::cubecl::logging::{LaunchObservation, LaunchObserver};
use burn::module::{Module, Param};
use burn::optim::{AdamWConfig, AdamWExecutionConfig, GradientsParams, Optimizer};
use burn::tensor::{DType, Device, Tensor, TensorData, Tolerance};
use std::sync::{Arc, Mutex};

const LR: f64 = 0.01;
const INITIAL: [[f32; 4]; 2] = [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]];
const GRADS: [[f32; 4]; 2] = [[0.5, -1.0, 2.0, -0.25], [-0.75, 0.125, -1.5, 3.0]];

#[derive(Default)]
struct Launches(Mutex<Vec<&'static str>>);

impl LaunchObserver for Launches {
    fn launched(&self, name: &'static str) {
        self.0.lock().unwrap().push(name);
    }
}

impl Launches {
    fn start() -> (Arc<Self>, LaunchObservation) {
        let launches = Arc::new(Self::default());
        let guard = LaunchObservation::new(launches.clone());
        (launches, guard)
    }

    fn check(&self, label: &str, batch: usize, single: usize) {
        let names = self.0.lock().unwrap();
        let actual_batch = names.iter().filter(|n| n.contains("adamw_foreach::")).count();
        let actual_single = names.iter()
            .filter(|n| n.contains("optim::adamw::eager::adamw::")).count();
        println!("{label}: batch={actual_batch} single={actual_single} total={}", names.len());
        println!("{label} launch_names={names:?}");
        assert_eq!((actual_batch, actual_single), (batch, single), "{label}: {names:?}");
    }
}

fn config(mode: AdamWExecutionConfig) -> AdamWConfig {
    AdamWConfig::new().with_beta_1(0.5).with_beta_2(0.5)
        .with_epsilon(1e-5).with_weight_decay(0.1).with_amsgrad(true)
        .with_execution(Some(mode))
}

fn tensor(values: [f32; 4], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
}

#[derive(Module, Debug)]
struct Pair {
    first: Param<Tensor<2>>,
    second: Param<Tensor<2>>,
}

impl Pair {
    fn new(device: &Device) -> Self {
        Self {
            first: Param::from_tensor(tensor(INITIAL[0], device)),
            second: Param::from_tensor(tensor(INITIAL[1], device)),
        }
    }

    fn params(&self) -> [&Param<Tensor<2>>; 2] {
        [&self.first, &self.second]
    }

    fn grads(&self, device: &Device, step: usize) -> GradientsParams {
        let mut result = GradientsParams::new();
        for (i, p) in self.params().into_iter().enumerate() {
            let scale = if step == 0 { 1.0 } else { -0.5 };
            result.register(p.id, tensor(GRADS[i].map(|g| g * scale), device));
        }
        result
    }
}

fn main() {
    let device = Device::cpu();
    device.sync().unwrap();
    // This public observer must see a known raw kernel before trusting batch counts.
    {
        let (launches, guard) = Launches::start();
        let output = config(AdamWExecutionConfig::Fused).build().step(
            LR, tensor(INITIAL[0], &device), tensor(GRADS[0], &device), None,
        );
        device.sync().unwrap();
        launches.check("CALIBRATION", 0, 1);
        drop(guard);
        drop(output);
    }
    let tracked = device.clone().autodiff();
    let mut actual = Pair::new(&tracked);
    let mut reference = actual.clone();
    let mut foreach = config(AdamWExecutionConfig::Foreach).init();
    let mut expected_optimizer = config(AdamWExecutionConfig::Reference).init();
    let mut scalar_params = INITIAL.map(|row| row.map(f64::from));
    let mut m1 = [[0.0_f64; 4]; 2];
    let mut m2 = [[0.0_f64; 4]; 2];
    let mut max_m2 = [[0.0_f64; 4]; 2];
    let mut max_error = 0.0_f64;
    for step in 0..2 {
        let expected_grads = reference.grads(&device, step);
        reference = expected_optimizer.step(LR, reference, expected_grads);
        let actual_grads = actual.grads(&device, step);
        device.sync().unwrap();
        {
            let (launches, guard) = Launches::start();
            actual = foreach.step(LR, actual, actual_grads);
            device.sync().unwrap();
            launches.check(&format!("FOREACH_STEP_{}", step + 1), 1, 0);
            drop(guard);
        }
        for (i, (a, r)) in actual.params().into_iter().zip(reference.params()).enumerate() {
            assert_eq!(a.id, r.id);
            assert_eq!(a.val().device(), tracked);
            assert!(a.val().is_require_grad());
            assert_eq!(a.val().dtype(), DType::F32);
            assert_eq!(a.val().dims(), [2, 2]);
            let data = a.val().to_data();
            r.val().to_data().assert_approx_eq::<f32>(&data, Tolerance::rel_abs(1e-5, 1e-6));
            let values = data.as_slice::<f32>().unwrap();
            for j in 0..4 {
                let g = f64::from(GRADS[i][j]) * if step == 0 { 1.0 } else { -0.5 };
                m1[i][j] = 0.5 * m1[i][j] + 0.5 * g;
                m2[i][j] = 0.5 * m2[i][j] + 0.5 * g * g;
                max_m2[i][j] = max_m2[i][j].max(m2[i][j]);
                let correction = 1.0 - 0.5_f64.powi((step + 1) as i32);
                let delta = (m1[i][j] / correction) / ((max_m2[i][j] / correction).sqrt() + 1e-5);
                scalar_params[i][j] = scalar_params[i][j] * (1.0 - LR * 0.1) - LR * delta;
                let error = (f64::from(values[j]) - scalar_params[i][j]).abs();
                max_error = max_error.max(error);
                assert!(error < 2e-6, "step={step} param={i} index={j} error={error}");
            }
            println!("VALUES step={} param={i} actual={values:?} scalar={:?}", step + 1, scalar_params[i]);
        }
    }
    println!("DOWNSTREAM_PASS steps=2 parameters=2 batch=2 single=0 max_scalar_error={max_error:e}");
}
'''


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Verification:
    def __init__(self, args):
        self.args = args
        self.run_dir = args.run_dir.resolve()
        require(self.run_dir.parent == (ROOT / "target").resolve()
                and self.run_dir.name.startswith("adamw-downstream-"),
                "--run-dir must be a direct target/adamw-downstream-* child")
        self.run_dir.mkdir(exist_ok=False)
        self.logs = self.run_dir / "logs"
        self.logs.mkdir()
        shutil.copyfile(__file__, self.run_dir / "verify.py")
        self.env = os.environ.copy()
        self.commands = []
        self.result = {
            "status": "running", "run_dir": str(self.run_dir),
            "commands_file": str(self.run_dir / "commands.json"),
            "scope": "Local git dependency structure only; no remote availability or PR3 completion claim.",
            "script_sha256": sha256(__file__), "executed_script_path": str(self.run_dir / "verify.py"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.result["previous_runs"] = []
        for previous in args.previous_run:
            previous = previous.resolve()
            require(previous.parent == (ROOT / "target").resolve()
                    and previous.name.startswith("adamw-downstream-"), "Invalid previous run directory")
            report = previous / "validation.json"
            data = json.loads(report.read_text())
            self.result["previous_runs"].append({
                "report": str(report), "sha256": sha256(report), "status": data["status"],
                "metadata_status": data.get("metadata", {}).get("status"),
                "build_status": data.get("build", {}).get("status"), "failure": data.get("failure"),
            })
        self.before = {p: sha256(ROOT / p) for p in ["Cargo.toml", "Cargo.lock"]}

    def command(self, argv, cwd=None, output=None, env=None, allow_failure=False):
        argv = [str(item) for item in argv]
        require("--config" not in argv, "Cargo --config is forbidden")
        number = len(self.commands) + 1
        stem = "{:03d}-{}".format(number, Path(argv[0]).name)
        stdout = output or self.logs / (stem + ".stdout.log")
        stderr = self.logs / (stem + ".stderr.log")
        cwd = Path(cwd or ROOT)
        entry = {"index": number, "argv": argv, "cwd": str(cwd), "stdout": str(stdout), "stderr": str(stderr)}
        self.commands.append(entry)
        start = time.monotonic()
        with Path(stdout).open("wb") as out, stderr.open("wb") as err:
            proc = subprocess.run(argv, cwd=cwd, env=env or self.env, stdout=out, stderr=err)
        entry.update(exit_code=proc.returncode, seconds=round(time.monotonic() - start, 3))
        entry.update(stdout_sha256=sha256(stdout), stderr_sha256=sha256(stderr))
        write_json(self.run_dir / "commands.json", self.commands)
        write_json(self.run_dir / "progress.json", {
            "status": self.result["status"], "completed_commands": number, "latest_command": entry,
        })
        if proc.returncode and not allow_failure:
            raise RuntimeError("Command failed ({}): {}; stderr: {}".format(proc.returncode, argv, stderr))
        return Path(stdout).read_text() if output is None else proc.returncode

    def configure(self):
        tools = {}
        for tool in ["cargo", "rustc", "rustdoc", "rustfmt"]:
            tools[tool] = self.command(["rustup", "which", "--toolchain", "beta", tool]).strip()
            require(Path(tools[tool]).is_absolute(), "Expected absolute beta tool path")
        removed = {}
        for key in list(self.env):
            if (key.startswith(("CARGO_", "RUST", "CUBECL_", "BURN_")) and key != "CARGO_HOME") or key == "DOCS_RS":
                removed[key] = self.env.pop(key)
        temp = self.run_dir / "tmp"
        temp.mkdir()
        self.env.update({
            "RUSTC": tools["rustc"], "RUSTDOC": tools["rustdoc"],
            "RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": "",
            "CARGO_BUILD_RUSTC_WRAPPER": "", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER": "",
            "CARGO_TARGET_DIR": str(self.run_dir / "build"), "CARGO_BUILD_JOBS": str(self.args.jobs),
            "CARGO_TERM_COLOR": "never", "BURN_DEVICE": "cpu", "TMPDIR": str(temp),
            "HOST_CC": "cc", "HOST_CXX": "c++", "CC": "cc", "CXX": "c++",
            "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_OPTIONAL_LOCKS": "0",
        })
        self.tools = tools
        self.result["toolchain"] = {
            name: {"path": path, "version": self.command([path, "--version"]).strip()}
            for name, path in tools.items()
        }
        self.result["environment"] = {k: self.env[k] for k in [
            "RUSTC", "RUSTDOC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            "CARGO_TARGET_DIR", "CARGO_BUILD_JOBS", "BURN_DEVICE", "TMPDIR", "HOST_CC", "HOST_CXX",
        ]}
        self.result["removed_environment_keys"] = sorted(removed)
        self.result["runner_environment"] = {k: v for k, v in self.env.items() if k.endswith("_RUNNER")}
        require(not self.result["runner_environment"], "Runner must be unset")
        self.result["identity"] = {
            key: self.command(["git", "config", "user." + key]).strip() for key in ["name", "email"]
        }
        require(all(self.result["identity"].values()), "Existing user git identity required")
        self.result["burn_head"] = self.command(["git", "rev-parse", "HEAD"]).strip()
        self.result["commit_date"] = self.command(
            ["git", "show", "-s", "--format=%cI", self.result["burn_head"]]).strip()
        self.result["inputs"] = {str(p): sha256(p) for p in PATCHES}
        self.result["inputs"].update({str(ROOT / p): h for p, h in self.before.items()})
        self.command(["git", "status", "--short"], output=self.logs / "main-status-before.log")

    def audit_configs(self, consumer):
        cargo_home = Path(self.env.get("CARGO_HOME", str(Path.home() / ".cargo")))
        configs = set()
        for base in [consumer] + list(consumer.parents):
            configs.update(base / ".cargo" / name for name in ["config", "config.toml"])
        configs.update(cargo_home / name for name in ["config", "config.toml"])
        self.result["cargo_configs"] = {}
        for path in sorted(configs):
            if not path.is_file():
                continue
            text = path.read_text()
            active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
            require(not re.search(r"(?m)^\s*\[\s*(patch|replace|source)(?:\.|\s*\])|^\s*(paths|runner)\s*=", active),
                    "Source override or runner in discovered Cargo config: " + str(path))
            self.result["cargo_configs"][str(path)] = {
                "sha256": sha256(path), "no_source_override": True, "no_configured_runner": True,
            }

    def cached_repo(self, name, revision):
        base = Path(self.env.get("CARGO_HOME", str(Path.home() / ".cargo"))) / "git"
        candidates = sorted((base / "db").glob(name + "-*"))
        candidates += sorted((base / "checkouts").glob(name + "-*/*"))
        for candidate in candidates:
            self.command(["git", "-C", candidate, "cat-file", "-e", revision + "^{commit}"], allow_failure=True)
            if self.commands[-1]["exit_code"] == 0:
                return candidate
        raise RuntimeError("Pinned {} {} is not cached".format(name, revision))

    def snapshot(self, name, source, revision, patches=(), dependencies=None):
        destination = self.run_dir / name
        destination.mkdir()
        archive = self.run_dir / (name + "-base.tar")
        self.command(["git", "-C", source, "archive", "--format=tar", revision], output=archive)
        self.command(["tar", "-xf", archive, "-C", destination])
        self.command(["git", "init", "-b", "main", destination])
        for patch in patches:
            self.command(["git", "apply", "--check", patch], cwd=destination)
            self.command(["git", "apply", patch], cwd=destination)
        manifest = destination / "Cargo.toml"
        text = manifest.read_text()
        for dependency, local in (dependencies or {}).items():
            pattern = r'(?m)^(' + re.escape(dependency) + r'\s*=\s*\{)([^\n]+)(\}\s*)$'
            def replace(match):
                body, git_count = re.subn(r'\bgit\s*=\s*"[^"]+"', 'git = ' + json.dumps(local["url"]), match[2])
                body, rev_count = re.subn(r'\brev\s*=\s*"[^"]+"', 'rev = ' + json.dumps(local["revision"]), body)
                require(git_count == rev_count == 1, "Unexpected dependency declaration: " + dependency)
                return match[1] + body + match[3]
            text, count = re.subn(pattern, replace, text)
            require(count == 1, "Expected exactly one workspace dependency: " + dependency)
        manifest.write_text(text)
        self.no_patch(manifest)
        commit_env = self.env.copy()
        for role in ["AUTHOR", "COMMITTER"]:
            commit_env["GIT_" + role + "_NAME"] = self.result["identity"]["name"]
            commit_env["GIT_" + role + "_EMAIL"] = self.result["identity"]["email"]
            commit_env["GIT_" + role + "_DATE"] = self.result["commit_date"]
        git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
        self.command(git + ["add", "--force", "."], cwd=destination)
        self.command(git + ["commit", "-m", "test(downstream): snapshot " + name + " for PR3 verification"],
                     cwd=destination, env=commit_env)
        local_rev = self.command(["git", "rev-parse", "HEAD"], cwd=destination).strip()
        require(not self.command(["git", "remote"], cwd=destination).strip(), "Unexpected remote")
        require(not self.command(["git", "status", "--porcelain"], cwd=destination).strip(), "Dirty snapshot")
        local_archive = self.run_dir / (name + "-local.tar")
        self.command(["git", "archive", "--format=tar", local_rev], cwd=destination, output=local_archive)
        item = {
            "cached_repo": str(source), "base_revision": revision, "path": str(destination),
            "url": destination.as_uri(), "revision": local_rev,
            "tree": self.command(["git", "rev-parse", "HEAD^{tree}"], cwd=destination).strip(),
            "archive": str(archive), "archive_sha256": sha256(archive),
            "local_archive": str(local_archive), "local_archive_sha256": sha256(local_archive),
            "manifest_sha256": sha256(manifest), "patches": [str(p) for p in patches],
        }
        self.result.setdefault("snapshots", {})[name] = item
        return item

    @staticmethod
    def no_patch(path):
        require(not re.search(r"(?m)^\s*\[\s*(patch|replace)(?:\.|\s*\])", path.read_text()),
                "Unexpected Cargo source override: " + str(path))

    def prepare(self):
        cubecl = self.snapshot("cubecl", self.cached_repo("cubecl", CUBECL_REV), CUBECL_REV, PATCHES[:1])
        cubek = self.snapshot("cubek", self.cached_repo("cubek", CUBEK_REV), CUBEK_REV, dependencies={
            name: cubecl for name in ["cubecl", "cubecl-common", "cubecl-environment", "cubecl-ir"]
        })
        deps = {name: cubecl for name in ["cubecl", "cubecl-common", "cubecl-environment", "cubecl-zspace"]}
        deps["cubek"] = cubek
        burn = self.snapshot("burn", ROOT, self.result["burn_head"], PATCHES[1:], deps)
        consumer = self.run_dir / "consumer"
        (consumer / "src").mkdir(parents=True)
        (consumer / "Cargo.toml").write_text('''[package]
name = "adamw-downstream-check"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]
resolver = "3"

[dependencies]
burn = { git = %s, rev = %s, default-features = false, features = ["std", "cpu", "optim", "cubecl"] }

[profile.dev]
debug = 0
strip = "none"
''' % (json.dumps(burn["url"]), json.dumps(burn["revision"])))
        (consumer / "src/main.rs").write_text(MAIN_RS)
        self.command([self.tools["rustfmt"], "--edition", "2024", consumer / "src/main.rs"])
        # CubeCL's default cache walks to the outermost Cargo.toml; confine it explicitly.
        (consumer / "cubecl.toml").write_text('[environment]\npath = { directory = %s }\n' %
                                             json.dumps(str(self.run_dir / "environment")))
        self.no_patch(consumer / "Cargo.toml")
        self.audit_configs(consumer)
        self.consumer = consumer

    def metadata(self):
        path = self.logs / "metadata.json"
        self.command([self.tools["cargo"], "metadata", "--format-version", "1",
                      "--manifest-path", self.consumer / "Cargo.toml"], cwd=self.consumer, output=path)
        data = json.loads(path.read_text())
        require(data["workspace_root"] == str(self.consumer), "Consumer must be an independent workspace")
        require(len(data["workspace_members"]) == 1, "Unexpected consumer workspace member")
        require(data["target_directory"] == self.env["CARGO_TARGET_DIR"], "Wrong target directory")
        packages = {p["id"]: p for p in data["packages"]}
        nodes = {n["id"]: n for n in data["resolve"]["nodes"]}
        expected = {name: "git+{url}?rev={revision}#{revision}".format(**item)
                    for name, item in self.result["snapshots"].items()}
        grouped = defaultdict(list)
        for package in packages.values():
            grouped[package["name"]].append(package)
        sources = {}
        exceptions = {}
        for name, group in sorted(grouped.items()):
            if name == "cubecl-hip-sys":
                require(all((p["source"] or "").startswith("registry+") for p in group), "hip-sys must be registry")
                exceptions[name] = [p["source"] for p in group]
                continue
            family = next((f for f in ["cubecl", "cubek", "burn"] if name == f or name.startswith(f + "-")), None)
            if family:
                require(len(group) == 1 and group[0]["source"] == expected[family],
                        "Duplicate or wrong source for " + name)
                sources[name] = group[0]["source"]
        for required in ["cubecl", "cubecl-runtime", "cubecl-cpu", "cubek", "burn", "burn-optim", "burn-cubecl"]:
            require(required in sources, "Missing required dependency " + required)
        edges = {"burn": [], "cubek": []}
        for family in edges:
            for package in packages.values():
                if package["source"] != expected[family]:
                    continue
                for dep in nodes[package["id"]]["deps"]:
                    target = packages[dep["pkg"]]
                    if target["source"] == expected["cubecl"]:
                        edges[family].append({"from": package["name"], "to": target["name"], "package_id": target["id"]})
        shared = sorted({e["package_id"] for e in edges["burn"]} & {e["package_id"] for e in edges["cubek"]})
        require(any(packages[p]["name"] == "cubecl" for p in shared), "Burn/CubeK must share the same CubeCL package ID")
        require(not any("fusion" in nodes[p["id"]]["features"] for p in packages.values()
                        if p["name"] in ["burn", "burn-cubecl", "burn-optim"]), "Raw CPU harness must disable fusion")
        cubecl_root = Path(self.result["snapshots"]["cubecl"]["path"])
        declared = []
        for manifest in sorted((cubecl_root / "crates").glob("*/Cargo.toml")):
            match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', manifest.read_text())
            if match:
                declared.append(match[1])
        graph = {
            "status": "passed", "packages": len(packages), "workspace_root": data["workspace_root"],
            "sources": sources, "registry_exceptions": exceptions, "direct_cubecl_edges": edges,
            "shared_cubecl_package_ids": shared,
            "cubecl_packages_in_graph": len([n for n in sources if n == "cubecl" or n.startswith("cubecl-")]),
            "cubecl_workspace_packages_absent_from_consumer_graph": sorted(set(declared) - set(grouped)),
            "no_root_patch": True, "no_source_config_override": True,
        }
        write_json(self.logs / "graph-summary.json", graph)
        self.result["metadata"] = {k: v for k, v in graph.items() if k not in ["sources", "direct_cubecl_edges"]}
        self.result["metadata"]["families"] = {
            family: {"source": source, "packages": sorted(n for n, s in sources.items() if s == source)}
            for family, source in expected.items()
        }
        self.result["metadata"]["direct_cubecl_edge_counts"] = {family: len(items) for family, items in edges.items()}
        self.save("metadata-passed")
        print("METADATA PASS: {} CubeCL packages; {} shared direct package IDs; {}".format(
            self.result["metadata"]["cubecl_packages_in_graph"], len(shared), self.run_dir), flush=True)

    def runtime(self):
        self.command([self.tools["cargo"], "build", "--locked", "--offline", "--manifest-path",
                      self.consumer / "Cargo.toml"], cwd=self.consumer)
        self.result["build"] = {"status": "passed", "command_index": self.commands[-1]["index"], "exit_code": 0}
        binary = self.run_dir / "build/debug/adamw-downstream-check"
        self.result["binary"] = {"path": str(binary), "sha256": sha256(binary)}
        output = self.command([binary], cwd=self.consumer)
        require("DOWNSTREAM_PASS steps=2 parameters=2 batch=2 single=0" in output, "Runtime success marker missing")
        self.result["runtime"] = {
            "status": "passed", "command_index": self.commands[-1]["index"], "exit_code": 0,
            "output": output,
        }
        print(output, flush=True)

    def save(self, status):
        self.result["status"] = status
        self.result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.result["report_script_sha256"] = sha256(__file__)
        # Historical target reports may retain raw config; published summaries never do.
        for config in self.result.get("cargo_configs", {}).values():
            config.pop("text", None)
        self.result["main_manifests_unchanged"] = {p: sha256(ROOT / p) == h for p, h in self.before.items()}
        require(all(self.result["main_manifests_unchanged"].values()), "Root manifest/lock changed during verification")
        artifacts = set(self.logs.glob("*")) | set(self.run_dir.glob("*.tar"))
        artifacts.update(self.run_dir / p for p in ["commands.json", "verify.py"])
        if hasattr(self, "consumer"):
            artifacts.update(self.consumer / p for p in ["Cargo.toml", "Cargo.lock", "src/main.rs", "cubecl.toml"])
        artifacts = sorted(p for p in artifacts if p.is_file())
        all_hashes = {str(p): sha256(p) for p in artifacts}
        self.result["artifacts_sha256"] = {
            path: value for path, value in all_hashes.items()
            if not path.endswith(".log") or Path(path).name == "failure.log"
        }
        write_json(self.run_dir / "validation.json", self.result)
        write_json(HERE / "validation.json", self.result)
        sums = [(sha256(__file__), Path(__file__)), (sha256(HERE / "validation.json"), HERE / "validation.json")]
        sums += [(value, Path(path)) for path, value in all_hashes.items()]
        if (HERE / "README.md").exists():
            sums.append((sha256(HERE / "README.md"), HERE / "README.md"))
        for previous in self.result["previous_runs"]:
            sums.append((previous["sha256"], Path(previous["report"])))
            prior = json.loads(Path(previous["report"]).read_text())
            sums.extend((digest, Path(path)) for path, digest in prior.get("artifacts_sha256", {}).items())
            if "binary" in prior:
                sums.append((prior["binary"]["sha256"], Path(prior["binary"]["path"])))
            for path in sorted((Path(previous["report"]).parent / "attempts").glob("*")):
                if path.is_file():
                    sums.append((sha256(path), path))
        if "binary" in self.result:
            sums.append((self.result["binary"]["sha256"], Path(self.result["binary"]["path"])))
        inventory = {str(path.relative_to(ROOT)): value for value, path in sums}
        (HERE / "SHA256SUMS").write_text("".join("{}  {}\n".format(value, path) for path, value in sorted(inventory.items())))

    def run(self):
        try:
            self.configure()
            self.prepare()
            self.metadata()
            if not self.args.metadata_only:
                self.runtime()
            self.command(["git", "status", "--short"], output=self.logs / "main-status-after.log")
            self.save("metadata-passed" if self.args.metadata_only else "passed")
        except BaseException as error:
            self.result["failure"] = {"type": type(error).__name__, "message": str(error)}
            (self.logs / "failure.log").write_text(traceback.format_exc())
            self.save("failed")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "target" / (
        "adamw-downstream-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    parser.add_argument("--jobs", type=int, default=2, help="Keep parallel build load modest")
    parser.add_argument("--metadata-only", action="store_true", help="Stop after graph validation")
    parser.add_argument("--previous-run", action="append", type=Path, default=[],
                        help="Include an earlier attempt's result/hash without overwriting its evidence")
    args = parser.parse_args()
    require(args.jobs > 0, "--jobs must be positive")
    Verification(args).run()


if __name__ == "__main__":
    main()
