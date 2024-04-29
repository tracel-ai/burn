//
// Note: If you are following the Burn Book guide this file can be ignored.
//
// This example file is added only for convenience and consistency so that
// the guide example can be executed like any other examples using:
//
//     cargo run --release --example guide
//
use std::process::Command;

fn main() {
    Command::new("cargo")
        .args(["run", "--bin", "guide"])
        .status()
        .expect("guide example should run");
}
