// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use std::path::{Path, PathBuf};
use std::{env, fs};

const PYTHON_PRINT_PYTORCH_DETAILS: &str = r"
import torch
from torch.utils import cpp_extension
print('LIBTORCH_VERSION:', torch.__version__.split('+')[0])
print('LIBTORCH_CXX11:', torch._C._GLIBCXX_USE_CXX11_ABI)
for include_path in cpp_extension.include_paths():
  print('LIBTORCH_INCLUDE:', include_path)
for library_path in cpp_extension.library_paths():
  print('LIBTORCH_LIB:', library_path)
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Os {
    Linux,
    Macos,
    Windows,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct SystemInfo {
    os: Os,
    cxx11_abi: String,
    libtorch_include_dirs: Vec<PathBuf>,
    libtorch_lib_dir: PathBuf,
}

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name)
}

impl SystemInfo {
    fn new() -> Option<Self> {
        let os = match env::var("CARGO_CFG_TARGET_OS")
            .expect("Unable to get TARGET_OS")
            .as_str()
        {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::Macos,
            os => panic!("unsupported TARGET_OS '{os}'"),
        };
        // Locate the currently active Python binary, similar to:
        // https://github.com/PyO3/maturin/blob/243b8ec91d07113f97a6fe74d9b2dcb88086e0eb/src/target.rs#L547
        let python_interpreter = match os {
            Os::Windows => PathBuf::from("python.exe"),
            Os::Linux | Os::Macos => {
                if env::var_os("VIRTUAL_ENV").is_some() {
                    PathBuf::from("python")
                } else {
                    PathBuf::from("python3")
                }
            }
        };
        let mut libtorch_include_dirs = vec![];
        let mut libtorch_lib_dir = None;
        let cxx11_abi = if env_var_rerun("LIBTORCH_USE_PYTORCH").is_ok() {
            let output = std::process::Command::new(&python_interpreter)
                .arg("-c")
                .arg(PYTHON_PRINT_PYTORCH_DETAILS)
                .output()
                .expect("error running python interpreter");
            let mut cxx11_abi = None;
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                match line.strip_prefix("LIBTORCH_CXX11: ") {
                    Some("True") => cxx11_abi = Some("1".to_owned()),
                    Some("False") => cxx11_abi = Some("0".to_owned()),
                    _ => {}
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_INCLUDE: ") {
                    libtorch_include_dirs.push(PathBuf::from(path))
                }
                if let Some(path) = line.strip_prefix("LIBTORCH_LIB: ") {
                    libtorch_lib_dir = Some(PathBuf::from(path))
                }
            }
            match cxx11_abi {
                Some(cxx11_abi) => cxx11_abi,
                None => panic!("no cxx11 abi returned by python {output:?}"),
            }
        } else {
            let libtorch = Self::prepare_libtorch_dir(os)?;
            let includes = env_var_rerun("LIBTORCH_INCLUDE")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            let lib = env_var_rerun("LIBTORCH_LIB")
                .map(PathBuf::from)
                .unwrap_or_else(|_| libtorch.clone());
            libtorch_include_dirs.push(includes.join("include"));
            libtorch_include_dirs.push(includes.join("include/torch/csrc/api/include"));
            if lib.ends_with("lib") {
                // DEP_TCH_LIBTORCH_LIB might already point to /lib
                libtorch_lib_dir = Some(lib);
            } else {
                libtorch_lib_dir = Some(lib.join("lib"));
            }
            env_var_rerun("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".to_owned())
        };
        let libtorch_lib_dir = libtorch_lib_dir?;
        Some(Self {
            os,
            cxx11_abi,
            libtorch_include_dirs,
            libtorch_lib_dir,
        })
    }

    fn check_system_location(os: Os) -> Option<PathBuf> {
        match os {
            Os::Linux => Path::new("/usr/lib/libtorch.so")
                .exists()
                .then(|| PathBuf::from("/usr")),
            _ => None,
        }
    }

    fn prepare_libtorch_dir(os: Os) -> Option<PathBuf> {
        if let Ok(libtorch) = env_var_rerun("DEP_TCH_LIBTORCH_LIB") {
            Some(PathBuf::from(libtorch))
        } else if let Ok(libtorch) = env_var_rerun("LIBTORCH") {
            Some(PathBuf::from(libtorch))
        } else if let Some(pathbuf) = Self::check_system_location(os) {
            Some(pathbuf)
        } else {
            check_out_dir()
        }
    }

    fn make(&self, use_cuda: bool, use_hip: bool) {
        let cuda_dependency = if use_cuda || use_hip {
            "src/cuda_hack/dummy_cuda_dependency.cpp"
        } else {
            "src/cuda_hack/fake_cuda_dependency.cpp"
        };
        println!("cargo:rerun-if-changed={}", cuda_dependency);

        match self.os {
            Os::Linux | Os::Macos => {
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .includes(&self.libtorch_include_dirs)
                    .flag(format!("-Wl,-rpath={}", self.libtorch_lib_dir.display()))
                    .flag("-std=c++17")
                    .flag(format!("-D_GLIBCXX_USE_CXX11_ABI={}", self.cxx11_abi))
                    .files(&[cuda_dependency])
                    .compile("burn-tch");
            }
            Os::Windows => {
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .includes(&self.libtorch_include_dirs)
                    .flag("/std:c++17")
                    .files(&[cuda_dependency])
                    .compile("burn-tch");
            }
        };
    }

    fn make_cpu() {
        let cuda_dependency = "src/cuda_hack/fake_cuda_dependency.cpp";
        println!("cargo:rerun-if-changed={}", cuda_dependency);

        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

        match os.as_str() {
            "windows" => {
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .flag("/std:c++17")
                    .files(&[cuda_dependency])
                    .compile("burn-tch");
            }
            _ => {
                cc::Build::new()
                    .cpp(true)
                    .pic(true)
                    .warnings(false)
                    .flag("-std=c++17")
                    .files(&[cuda_dependency])
                    .compile("tch");
            }
        };
    }
}

fn check_out_dir() -> Option<PathBuf> {
    let out_dir = env_var_rerun("OUT_DIR").ok()?;
    let libtorch_dir = PathBuf::from(out_dir).join("libtorch");
    libtorch_dir.exists().then_some(libtorch_dir)
}

fn main() {
    let system_info = SystemInfo::new();
    let out_dir = env_var_rerun("OUT_DIR").expect("Failed to get out dir");

    let mut gpu_found = false;
    let found_dir = system_info.is_some();
    if let Some(system_info) = &system_info {
        let si_lib = &system_info.libtorch_lib_dir;
        let use_cuda =
            si_lib.join("libtorch_cuda.so").exists() || si_lib.join("torch_cuda.dll").exists();
        let use_hip =
            si_lib.join("libtorch_hip.so").exists() || si_lib.join("torch_hip.dll").exists();

        system_info.make(use_cuda, use_hip);
        gpu_found = use_cuda || use_hip;
    } else {
        SystemInfo::make_cpu();
    }
    let check_file = PathBuf::from(out_dir).join("tch_gpu_check.rs");
    if gpu_found {
        fs::write(check_file, "#[allow(clippy::no_effect)]\n()").unwrap();
    } else {
        let message = if !found_dir {
            r#"Could not find libtorch dir.

        If you are trying to use the automatically downloaded version, the path is not directly available on Windows. Instead, try setting the `LIBTORCH` environment variable for the manual download instructions.

        If the library has already been downloaded in the torch-sys OUT_DIR, you can point the variable to this path (or move the downloaded lib and point to it)."#
        } else {
            "No libtorch_cuda or libtorch_hip found. Download the GPU version of libtorch to use a GPU device"
        };
        fs::write(check_file, format!("panic!(\"{message}\")")).unwrap();
    }
}
