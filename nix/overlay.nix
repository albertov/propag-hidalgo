inputs: final: prev:
let

  rust_cuda_sha256 = "sha256-3cpFOdAdoKLnd5HB9ryNWIUOXlF/g1cm8RA+0nAQDK0=";
  cargoDepsHash = "sha256-NLY3TkUO5Kz0tvA2z2tXeDmeTKGFoC3cDa8Vg6uRNBM=";
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
  CUDA_ROOT = final.cudaCombined;
  CUDA_PATH = final.cudaCombined;
  LLVM_CONFIG = "${final.llvmPackages_7.llvm.dev}/bin/llvm-config";
  LLVM_LINK_SHARED = "1";

  workspaceArgs = {
    src = final.lib.cleanSource ../crates;
    cargoDeps = final.myRustPlatform.importCargoLock {
      lockFile = ../crates/Cargo.lock;
      outputHashes = {
        "const_soft_float-0.1.4" = "sha256-fm2e3np+q4yZjAafkwbxTqUZBgVDrQ/l4hxMD+l7kMA=";
        "cuda_builder-0.3.0" = rust_cuda_sha256;
        "cuda_std-0.2.2" = rust_cuda_sha256;
      };
    };
    inherit
      BINDGEN_EXTRA_CLANG_ARGS
      LIBCLANG_PATH
      CUDA_PATH
      LLVM_CONFIG
      CUDA_ROOT
      LLVM_LINK_SHARED
      ;
    # Remove unneeded references (propably in the embedded PTX) to massively
    # reduce closure size from ~7Gb to ~200M
    fixupPhase = with final; ''
      if [[ -d $out/bin ]]; then
        find $out/bin -type f -exec \
          remove-references-to \
            -t ${myRustToolchain} \
            -t ${cudaPackages.backendStdenv.cc} \
            -t ${cudaPackages.cuda_nvcc} \
            -t ${cudaCombined} \
            {} \;
      fi
    '';
  };
in
{

  propag = final.callPackage ./pkgs/propag.nix {
    # Otherwise deb package is huge
    gdal = final.gdal-small;
  };

  py-propag = final.callPackage ./pkgs/py-propag.nix {
    # Otherwise deb package is huge
    gdal = final.gdal-small;
  };

  buildWorkspacePythonPackage =
    args:
    final.python3.pkgs.buildPythonPackage (
      workspaceArgs
      // args
      // {
        nativeBuildInputs = args.nativeBuildInputs or [ ] ++ [
          final.myRustToolchain
          final.removeReferencesTo
        ];
        build-system = with final.myRustPlatform; [
          cargoSetupHook
          maturinBuildHook
        ];
      }
    );

  buildWorkspacePackage =
    args:
    final.myRustPlatform.buildRustPackage (
      workspaceArgs
      // args
      // {
        nativeBuildInputs = args.nativeBuildInputs or [ ] ++ [
          final.myRustToolchain
          final.removeReferencesTo
        ];
      }
    );

  readCargo = x: builtins.fromTOML (builtins.readFile x);

  gdal-small = final.callPackage ./gdal.nix { };

  gdal = prev.gdal.override {
    useMinimalFeatures = true;
    usePostgres = true;
  };

  cudaPackages = prev.cudaPackages_12;

  cudaCombined = final.symlinkJoin {
    name = "cuda-combined";
    paths = with final; [
      cudaPackages.cuda_cudart
      cudaPackages.cuda_cudart.dev
      cudaPackages.cuda_cudart.static
      cudaPackages.cuda_nvprof
      cudaPackages.cuda_cccl
      cudaPackages.cuda_cccl.dev
      cudaPackages.cuda_nvcc
      cudaPackages.cuda_profiler_api.dev
      #cudaPackages.cuda_cupti
    ];
  };

  llvmPackages_7 =
    with final;
    with lib;
    recurseIntoAttrs (
      callPackage "${inputs.nixpkgs_old}/pkgs/development/compilers/llvm/7" {
        inherit (stdenvAdapters) overrideCC;
        buildLlvmTools = buildPackages.llvmPackages_7.tools;
        targetLlvm = targetPackages.llvmPackages_7.llvm or llvmPackages_7.llvm;
        targetLlvmLibraries = targetPackages.llvmPackages_7.libraries or llvmPackages_7.libraries;
      }
    );

  myRustToolchain = final.buildPackages.rust-bin.fromRustupToolchainFile ../crates/rust-toolchain.toml;

  myRustPlatform = final.buildPackages.makeRustPlatform {
    stdenv = final.cudaPackages.backendStdenv;
    cargo = final.myRustToolchain;
    rustc = final.myRustToolchain;
  };

}
