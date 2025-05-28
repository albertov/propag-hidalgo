final: _prev:
let
  inherit (final) lib buildPackages stdenv;
  libclang = buildPackages.llvmPackages.libclang.lib;
  clangMajorVer = builtins.head (lib.splitString "." (lib.getVersion buildPackages.clang));
in
{
  myRustToolchain = final.buildPackages.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
  myRustPlatform = final.buildPackages.makeRustPlatform {
    cargo = final.myRustToolchain;
    rustc = final.myRustToolchain;
  };

  firelib-rs = final.myRustPlatform.buildRustPackage {
    pname = "firelib-rs";
    version = "git";
    src = final.lib.cleanSource ./.;
    cargoLock.lockFile = ./Cargo.lock;
    cargoLock.outputHashes = {
      "const_soft_float-0.1.4" = "sha256-fm2e3np+q4yZjAafkwbxTqUZBgVDrQ/l4hxMD+l7kMA=";
    };
    buildAndTestSubdir = "firelib-rs";
    BINDGEN_EXTRA_CLANG_ARGS = ''
      -isystem ${libclang}/lib/clang/${clangMajorVer}/include
      ${builtins.readFile "${stdenv.cc}/nix-support/libc-crt1-cflags"}
      ${builtins.readFile "${stdenv.cc}/nix-support/libc-cflags"}
      ${builtins.readFile "${stdenv.cc}/nix-support/cc-cflags"}
      ${builtins.readFile "${stdenv.cc}/nix-support/libcxx-cxxflags"}
    '';
    LIBCLANG_PATH = "${libclang}/lib";
  };
}
