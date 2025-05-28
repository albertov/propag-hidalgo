use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new(".")
        .copy_to("lala.ptx")
        .build()
        .unwrap();
}
