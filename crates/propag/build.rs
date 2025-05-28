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
    CudaBuilder::new("../firelib")
        .copy_to(dest)
        .build()
        .unwrap();

    let dest = format!("{}/propag_host.h", include_dir);
    println!("cargo::rerun-if-changed={}", dest);
    cbindgen::Builder::new()
        .with_crate(".")
        .with_after_include(
            "
    #include \"geometry.h\"
    typedef float T;
    ",
        )
        .with_include_guard("PROPAG_HOST_H")
        //.with_parse_deps(true)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&dest);

    let dest = format!("{}/geometry.h", include_dir);
    println!("cargo::rerun-if-changed={}", dest);
    cbindgen::Builder::new()
        .with_crate("../geometry")
        .with_parse_deps(true)
        .with_include_guard("GEOMETRY_H")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(&dest);

    let dest = format!("{}/propag_c.ptx", target_dir);
    println!("cargo::rerun-if-changed={}", dest);
    Command::new("nvcc")
        .args([
            "-arch",
            "compute_62",
            //"-use_fast_math",
            "-O3",
            //"--restrict",
            //"--expt-relaxed-constexpr",
            "-ptx",
            "-I",
            &include_dir,
            "-I",
            "./src",
            "-o",
            &dest,
            "src/propag.cu",
        ])
        .status()
        .expect("failed to find nvcc")
        .exit_ok()
        .expect("failed to execute nvcc");
}
