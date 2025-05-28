final: _prev:
let
  inherit (final) lib buildPackages stdenv;
  libclang = buildPackages.llvmPackages.libclang.lib;
  clangMajorVer = builtins.head (lib.splitString "." (lib.getVersion buildPackages.clang));
  BINDGEN_EXTRA_CLANG_ARGS = ''
    -isystem ${libclang}/lib/clang/${clangMajorVer}/include
    ${builtins.readFile "${stdenv.cc}/nix-support/libc-crt1-cflags"}
    ${builtins.readFile "${stdenv.cc}/nix-support/libc-cflags"}
    ${builtins.readFile "${stdenv.cc}/nix-support/cc-cflags"}
    ${builtins.readFile "${stdenv.cc}/nix-support/libcxx-cxxflags"}
  '';
  LIBCLANG_PATH = "${libclang}/lib";
  CUDA_PATH=final.cudatoolkit;
  LLVM_CONFIG = "${final.llvmPackages_7.llvm.dev}/bin/llvm-config";

  workspaceArgs = {
    version = "git";
    src = final.lib.cleanSource ./.;
    cargoLock.lockFile = ./Cargo.lock;
    cargoLock.outputHashes = {
      "const_soft_float-0.1.4" = "sha256-fm2e3np+q4yZjAafkwbxTqUZBgVDrQ/l4hxMD+l7kMA=";
    };
    inherit BINDGEN_EXTRA_CLANG_ARGS LIBCLANG_PATH CUDA_PATH LLVM_CONFIG;
  };
in
{
  inherit (final.pkgs_2311) llvmPackages_7;

  myRustToolchain = (final.buildPackages.rust-bin.fromRustupToolchainFile
  ./rust-toolchain.toml).override {
    extensions = ["rust-src" "rustc-dev" "llvm-tools-preview"];
  };
  myRustPlatform = final.buildPackages.makeRustPlatform {
    cargo = final.myRustToolchain;
    rustc = final.myRustToolchain;
    rustc-src = final.myRustToolchain;
    rustc-dev = final.myRustToolchain;
    llvm-tools-preview = final.myRustToolchain;
  };


  firelib-rs = final.myRustPlatform.buildRustPackage (workspaceArgs // {
    pname = "firelib-rs";
    buildAndTestSubdir = "firelib-rs";
  });

  firelib-cuda = final.myRustPlatform.buildRustPackage (workspaceArgs // {
    pname = "firelib-cuda";
    buildAndTestSubdir = "firelib-cuda";
    buildInputs = with final; with final.myRustToolchain; [
      cudatoolkit
      cudatoolkit.lib
      openssl
    ];
    nativeBuildInputs = with final; [
      pkg-config
      myRustToolchain
    ];

  });
}
