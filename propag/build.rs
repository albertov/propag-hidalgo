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
}
