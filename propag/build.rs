#![feature(exit_status_error)]
use std::process::Command;
use cuda_builder::CudaBuilder;

fn main() {
    let target_dir = "../target/cuda/";
    let dest = format!("{}/firelib.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    std::fs::create_dir_all(target_dir).unwrap();
    CudaBuilder::new("../firelib-cuda")
        .copy_to(dest)
        .build()
        .unwrap();

    let dest = format!("{}/propag_c.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    Command::new("nvcc")
        .args(["-ptx", "-o", &dest, "src/propag.cu"])
        .status()
        .expect("failed to find nvcc")
        .exit_ok()
        .expect("failed to execute nvcc");
}
