use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("/home/alberto/src/propag25/firelib-cuda")
        .build()
        .unwrap();
}
