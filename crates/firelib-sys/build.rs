use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=fireLib.c");
    println!("cargo::rerun-if-changed=fireLib.h");
    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("fireLib.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_file("fireLib.h")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    cc::Build::new()
        .file("fireLib.c")
        // Disable warnings because we don't care
        .flag("-w")
        .opt_level(3)
        .compile("fireLib");
}
