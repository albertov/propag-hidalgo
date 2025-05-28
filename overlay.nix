final: _prev: {
  rustToolchain = final.buildPackages.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
  rustPlatform = final.buildPackages.makeRustPlatform {
    cargo = final.rustToolchain;
    rustc = final.rustToolchain;
  };

  firelib-rs = final.rustPlatform.buildRustPackage {
    pname = "firelib-rs";
    version = "git";
    src = final.lib.cleanSource ./.;
    cargoLock.lockFile = ./Cargo.lock;
    cargoLock.outputHashes = {
      "const_soft_float-0.1.4" = "sha256-fm2e3np+q4yZjAafkwbxTqUZBgVDrQ/l4hxMD+l7kMA=";
    };
    buildAndTestSubdir = "firelib-rs";
    BINDGEN_EXTRA_CLANG_ARGS = with final; ''
      $(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
      $(< ${stdenv.cc}/nix-support/libc-cflags) \
      $(< ${stdenv.cc}/nix-support/cc-cflags) \
      $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
      ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
      ${lib.optionalString stdenv.cc.isGNU "-isystem
      ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}
      -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -isystem ${buildPackages.llvmPackages.libclang.lib}/lib/clang/${builtins.head (lib.splitString "." (lib.getVersion buildPackages.clang))}/include"}
    '';
    LIBCLANG_PATH = "${final.buildPackages.llvmPackages.libclang.lib}/lib";
  };
}
