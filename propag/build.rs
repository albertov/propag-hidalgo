use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../firelib-cuda")
        .copy_to("../lala.ptx")
        .build()
        .unwrap();
}
