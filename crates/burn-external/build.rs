fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::metadata=placeholder=1");
}
