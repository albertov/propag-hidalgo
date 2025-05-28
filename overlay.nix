inputs: final: _prev:
let
  inherit (final) lib buildPackages stdenv;
  libclang = buildPackages.llvmPackages.libclang.lib;
  clangMajorVer = builtins.head (lib.splitString "." (lib.getVersion buildPackages.clang));
  BINDGEN_EXTRA_CLANG_ARGS = ''
    -isystem ${libclang}/lib/clang/${clangMajorVer}/include \
    ${builtins.readFile "${stdenv.cc}/nix-support/libc-crt1-cflags"} \
    ${builtins.readFile "${stdenv.cc}/nix-support/libc-cflags"} \
    ${builtins.readFile "${stdenv.cc}/nix-support/cc-cflags"} \
    ${builtins.readFile "${stdenv.cc}/nix-support/libcxx-cxxflags"}
  '';
  LIBCLANG_PATH = "${libclang}/lib";
  CUDA_PATH=final.cudatoolkit_11;
  CUDA_ROOT=final.cudatoolkit_11;
  LLVM_CONFIG = "${final.llvmPackages_7.llvm.dev}/bin/llvm-config";
  LLVM_LINK_SHARED="1";

  workspaceArgs = {
    version = "git";
    src = final.lib.cleanSource ./.;
    cargoLock.lockFile = ./Cargo.lock;
    cargoLock.outputHashes = {
      "const_soft_float-0.1.4" = "sha256-fm2e3np+q4yZjAafkwbxTqUZBgVDrQ/l4hxMD+l7kMA=";
      "cuda_builder-0.3.0" = "sha256-5LZiNLgGg5gweabC0cfJ9Hbxpu98JIIehjuoNUh4aLI=";
      "cuda_std-0.2.2" = "sha256-5LZiNLgGg5gweabC0cfJ9Hbxpu98JIIehjuoNUh4aLI=";
    };
    inherit BINDGEN_EXTRA_CLANG_ARGS LIBCLANG_PATH CUDA_PATH LLVM_CONFIG
    CUDA_ROOT;
  };
in
{
  firelib-rs = final.myRustPlatform.buildRustPackage (workspaceArgs // {
    pname = "firelib-rs";
    buildAndTestSubdir = "firelib-rs";
  });

  propag = final.myRustPlatform.buildRustPackage (workspaceArgs // {
    pname = "propag";
    buildAndTestSubdir = "propag";
    buildInputs = with final; with final.myRustToolchain.availableComponents; [
      cudatoolkit_11.lib
      openssl
    ];
    LLVM_LINK_SHARED = 1;
    nativeBuildInputs = with final; [
      cudatoolkit_11
      pkg-config
      myRustToolchain
      llvmPackages_7.llvm
      ncurses # nvmm backend needs it
      which
    ];

  });


  llvmPackages_7 = with final; with lib;
  recurseIntoAttrs (callPackage "${inputs.nixpkgs_old}/pkgs/development/compilers/llvm/7" 
  {
    inherit (stdenvAdapters) overrideCC;
    buildLlvmTools = buildPackages.llvmPackages_7.tools;
    targetLlvm = targetPackages.llvmPackages_7.llvm or llvmPackages_7.llvm;
    targetLlvmLibraries = targetPackages.llvmPackages_7.libraries or llvmPackages_7.libraries;
  });

  myRustToolchain =
    final.buildPackages.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

  myRustPlatform = final.buildPackages.makeRustPlatform {
    cargo = final.myRustToolchain;
    rustc = final.myRustToolchain;
  };


}
