use cuda_builder::CudaBuilder;


fn main() {
    let target_dir = "../target/cuda/";
    std::fs::create_dir_all(target_dir).unwrap();
    CudaBuilder::new("../firelib-cuda")
        .copy_to(format!("{}/firelib.ptx", target_dir))
        .build()
        .unwrap();
}
