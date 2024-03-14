#[cfg(all(feature = "dawn", feature = "wgpu"))]
compile_error!(
    "The 'dawn' and 'wgpu' backends can not be used at the same time, you must choose one."
);
#[cfg(all(feature = "dawn", not(target_os = "macos")))]
compile_error!("The 'dawn' backend currently only builds on macos.");

fn main() {
    #[cfg(feature = "dawn")]
    link_and_bind_dawn();
}

#[cfg(feature = "dawn")]
fn link_and_bind_dawn() {
    use bindgen::builder;
    use std::env;
    use std::path::PathBuf;

    let dawn_src_dir = "dawn";

    let repo = match git2::Repository::open("../..") {
        Ok(repo) => repo,
        Err(err) => panic!("failed to open repo: {err}"),
    };
    let mut submodules = match repo.submodules() {
        Ok(submodules) => submodules,
        Err(err) => panic!("failed to list git submodules: {err}"),
    };
    for submodule in submodules.iter_mut() {
        if submodule.name().unwrap().ends_with("dawn") {
            match submodule.update(true, None) {
                Ok(_) => (),
                Err(err) => panic!("failed to update the dawn submodule: {err}"),
            }
        }
    }

    std::process::Command::new("python3")
        .current_dir(dawn_src_dir)
        .arg("tools/fetch_dawn_dependencies.py")
        .arg("--use-test-deps")
        .spawn()
        .expect("failed to fetch Dawn dependencies");

    let dst = cmake::Config::new(dawn_src_dir)
        .profile("Release")
        .build_arg("-j8")
        .build();

    let dawn_build_dir = dst.join("build");
    let dawn_build_dir = dawn_build_dir.display();

    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=Cocoa");
    println!("cargo:rustc-link-search={dawn_build_dir}/src/dawn");
    println!("cargo:rustc-link-lib=static=dawn_headers");
    println!("cargo:rustc-link-lib=static=dawn_proc");
    println!("cargo:rustc-link-lib=static=dawncpp");
    println!("cargo:rustc-link-lib=static=dawncpp_headers");
    println!("cargo:rustc-link-search={dawn_build_dir}/src/dawn/common");
    println!("cargo:rustc-link-lib=static=dawn_common");
    println!("cargo:rustc-link-search={dawn_build_dir}/src/dawn/native");
    println!("cargo:rustc-link-lib=static=dawn_native");
    println!("cargo:rustc-link-search={dawn_build_dir}/src/dawn/platform");
    println!("cargo:rustc-link-lib=static=dawn_platform");
    println!("cargo:rustc-link-search={dawn_build_dir}/src/tint");
    println!("cargo:rustc-link-lib=static=tint_api");
    println!("cargo:rustc-link-lib=static=tint_api_common");
    println!("cargo:rustc-link-lib=static=tint_api_options");
    println!("cargo:rustc-link-lib=static=tint_cmd_common");
    println!("cargo:rustc-link-lib=static=tint_lang_core");
    println!("cargo:rustc-link-lib=static=tint_lang_core_constant");
    println!("cargo:rustc-link-lib=static=tint_lang_core_intrinsic");
    println!("cargo:rustc-link-lib=static=tint_lang_core_ir");
    println!("cargo:rustc-link-lib=static=tint_lang_core_ir_transform");
    println!("cargo:rustc-link-lib=static=tint_lang_core_type");
    println!("cargo:rustc-link-lib=static=tint_lang_glsl_validate");
    println!("cargo:rustc-link-lib=static=tint_lang_hlsl_writer_common");
    println!("cargo:rustc-link-lib=static=tint_lang_msl");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_intrinsic");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_ir");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_validate");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_ast_printer");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_ast_raise");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_common");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_helpers");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_printer");
    println!("cargo:rustc-link-lib=static=tint_lang_msl_writer_raise");
    println!("cargo:rustc-link-lib=static=tint_lang_spirv");
    println!("cargo:rustc-link-lib=static=tint_lang_spirv_intrinsic");
    println!("cargo:rustc-link-lib=static=tint_lang_spirv_ir");
    println!("cargo:rustc-link-lib=static=tint_lang_spirv_reader_lower");
    println!("cargo:rustc-link-lib=static=tint_lang_spirv_type");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_ast");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_ast_transform");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_common");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_features");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_helpers");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_inspector");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_intrinsic");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_ir");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_program");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_reader");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_reader_lower");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_reader_parser");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_reader_program_to_ir");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_resolver");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_sem");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_writer");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_writer_ast_printer");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_writer_ir_to_program");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_writer_raise");
    println!("cargo:rustc-link-lib=static=tint_lang_wgsl_writer_syntax_tree_printer");
    println!("cargo:rustc-link-lib=static=tint_utils_bytes");
    println!("cargo:rustc-link-lib=static=tint_utils_cli");
    println!("cargo:rustc-link-lib=static=tint_utils_command");
    println!("cargo:rustc-link-lib=static=tint_utils_containers");
    println!("cargo:rustc-link-lib=static=tint_utils_debug");
    println!("cargo:rustc-link-lib=static=tint_utils_diagnostic");
    println!("cargo:rustc-link-lib=static=tint_utils_file");
    println!("cargo:rustc-link-lib=static=tint_utils_generator");
    println!("cargo:rustc-link-lib=static=tint_utils_ice");
    println!("cargo:rustc-link-lib=static=tint_utils_id");
    println!("cargo:rustc-link-lib=static=tint_utils_macros");
    println!("cargo:rustc-link-lib=static=tint_utils_math");
    println!("cargo:rustc-link-lib=static=tint_utils_memory");
    println!("cargo:rustc-link-lib=static=tint_utils_reflection");
    println!("cargo:rustc-link-lib=static=tint_utils_result");
    println!("cargo:rustc-link-lib=static=tint_utils_rtti");
    println!("cargo:rustc-link-lib=static=tint_utils_socket");
    println!("cargo:rustc-link-lib=static=tint_utils_strconv");
    println!("cargo:rustc-link-lib=static=tint_utils_symbol");
    println!("cargo:rustc-link-lib=static=tint_utils_text");
    println!("cargo:rustc-link-lib=static=tint_utils_traits");
    println!("cargo:rustc-link-search={dawn_build_dir}/third_party/abseil/absl/strings");
    println!("cargo:rustc-link-lib=static=absl_str_format_internal");
    println!("cargo:rustc-link-lib=static=absl_strings");
    println!("cargo:rustc-link-lib=static=absl_strings_internal");
    println!("cargo:rustc-link-search={dawn_build_dir}/third_party/abseil/absl/base");
    println!("cargo:rustc-link-lib=static=absl_base");
    println!("cargo:rustc-link-lib=static=absl_spinlock_wait");
    println!("cargo:rustc-link-lib=static=absl_throw_delegate");
    println!("cargo:rustc-link-lib=static=absl_raw_logging_internal");
    println!("cargo:rustc-link-lib=static=absl_log_severity");
    println!("cargo:rustc-link-search={dawn_build_dir}/third_party/abseil/absl/numeric");
    println!("cargo:rustc-link-lib=static=absl_int128");
    println!("cargo:rustc-link-search={dawn_build_dir}/third_party/abseil/absl/hash");
    println!("cargo:rustc-link-lib=static=absl_city");
    println!("cargo:rustc-link-lib=static=absl_hash");
    println!("cargo:rustc-link-lib=static=absl_low_level_hash");
    println!("cargo:rustc-link-search={dawn_build_dir}/third_party/abseil/absl/container");
    println!("cargo:rustc-link-lib=static=absl_hashtablez_sampler");
    println!("cargo:rustc-link-lib=static=absl_raw_hash_set");
    println!("cargo:rustc-link-lib=dylib=c++");

    let bindings = builder()
        .header("dawn.h")
        .clang_args([
            "-x",
            "c++",
            "-I",
            std::format!("{dawn_build_dir}/gen/include/").as_str(),
            "-I",
            std::format!("{dawn_src_dir}/include/").as_str(),
            "--std=c++17",
        ])
        .allowlist_function(".*GetProcs.*")
        .allowlist_function(".*SetProcs.*")
        .allowlist_function("wgpu.*")
        .allowlist_file(".*webgpu.h")
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate Dawn bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("dawn_native_bindings_gen.rs"))
        .expect("Couldn't write Dawn bindings!");
}
