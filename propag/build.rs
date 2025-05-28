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

    let dest = format!("{}/firelib_cuda.h", target_dir);
    cbindgen::Builder::new()
      .with_crate("../firelib-cuda")
      .with_language(cbindgen::Language::C)
      .with_parse_deps(true)
      .generate()
      .expect("Unable to generate bindings")
      .write_to_file(&dest);


    let dest = format!("{}/propag_c.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    Command::new("nvcc")
        .args(["-ptx", "-o", &dest, "src/propag.cu"])
        .status()
        .expect("failed to find nvcc")
        .exit_ok()
        .expect("failed to execute nvcc");
}
