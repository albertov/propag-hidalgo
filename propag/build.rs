#![feature(exit_status_error)]
use cuda_builder::CudaBuilder;
use std::process::Command;

fn main() {
    let target_dir = "../target/cuda/";
    std::fs::create_dir_all(target_dir).unwrap();
    let include_dir = format!("{}/include", target_dir);
    std::fs::create_dir_all(&include_dir).unwrap();

    let dest = format!("{}/firelib.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    CudaBuilder::new("../firelib-cuda")
        .copy_to(dest)
        .build()
        .unwrap();

    let dest = format!("{}/firelib_cuda.h", include_dir);
    println!("cargo::rerun-if-changed={}", dest);
    cbindgen::Builder::new()
        .with_crate("../firelib-cuda")
        .with_language(cbindgen::Language::C)
        .with_after_include("
        #define T float
        #define Max_MAX SIZE_MAX
        ")
        //.with_parse_deps(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&dest);

    let dest = format!("{}/geometry.h", include_dir);
    println!("cargo::rerun-if-changed={}", dest);
    cbindgen::Builder::new()
        .with_crate("../geometry")
        .with_language(cbindgen::Language::C)
        .with_parse_deps(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&dest);

    let dest = format!("{}/propag_c.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    Command::new("nvcc")
        .args([
            "-arch",
            "compute_62",
            "-ptx",
            "-I",
            &include_dir,
            "-o",
            &dest,
            "src/propag.cu",
        ])
        .status()
        .expect("failed to find nvcc")
        .exit_ok()
        .expect("failed to execute nvcc");
}
