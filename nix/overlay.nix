self: inputs: system: final: prev:
let

  rust_cuda_sha256 = "sha256-3cpFOdAdoKLnd5HB9ryNWIUOXlF/g1cm8RA+0nAQDK0=";
  cargoDepsHash = "sha256-NLY3TkUO5Kz0tvA2z2tXeDmeTKGFoC3cDa8Vg6uRNBM=";

  inherit (final) lib buildPackages stdenv;
  libclang = buildPackages.llvmPackages.libclang.lib;

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
  fpm =
    builtins.head
      (inputs.nix-utils.rpmDebUtils.${system}.buildFakeSingleDeb null null).buildInputs;

  libecwj2 = final.callPackage ./pkgs/libecwj2 { };

  qgis-propag-algo = final.callPackage ./pkgs/qgis-propag-algo.nix { };

  qgis = prev.qgis.override {
    extraPythonPackages =
      ps: with ps; [
        ipython
        jupyter
        qtconsole
      ];
  };

  qgis-debug = prev.qgis.unwrapped.overrideAttrs (old: {
    cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Debug" ] ++ old.cmakeFlags;
    meta = old.meta // {
      mainProgram = "qgis";
    };
  });
  geometry = final.callPackage ./pkgs/geometry.nix { };

  propag = final.callPackage ./pkgs/propag.nix {
    # Otherwise deb package is huge
    # gdal = final.gdal-small;
  };

  py-propag = final.callPackage ./pkgs/py-propag.nix {
    # Otherwise deb package is huge
    # gdal = final.gdal-small;
  };

  buildWorkspacePythonPackage =
    args:
    final.python3.pkgs.buildPythonPackage (
      workspaceArgs
      // args
      // {
        nativeBuildInputs = args.nativeBuildInputs or [ ] ++ [
          final.myRustPlatform.bindgenHook
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
          final.myRustPlatform.bindgenHook
          final.myRustToolchain
          final.removeReferencesTo
        ];
      }
    );

  readCargo = x: builtins.fromTOML (builtins.readFile x);

  gdal-small = final.callPackage ./gdal.nix { };

  #gdal = final.gdal-small;

  gdal =
    (prev.gdal.override {
      useMinimalFeatures = true;
      usePostgres = true;
    }).overrideAttrs
      (old: {
        cmakeFlags = old.cmakeFlags ++ [ "-DECW_ROOT=${final.libecwj2}" ];
        disabledTests = old.disabledTests ++ [ "test_ecw_online_6" ];

      });

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
  clippy = final.myRustPlatform.rust.rustc.availableComponents.clippy;

  pre-commit = inputs.pre-commit-nix.lib.${system}.run rec {
    src = self;
    hooks = {
      treefmt-nix.enable = true;
      treefmt-nix.entry = "${final.nix}/bin/nix fmt";
      clippy.enable = false; # FIXME
      clippy.packageOverrides.clippy = final.clippy;
      clippy.packageOverrides.cargo = final.myRustToolchain;
    };
  };

  ubuntize =
    drv:
    final.runCommandLocal "ubuntize"
      {
        nativeBuildInputs = [
          final.patchelf
          final.rsync
          final.binutils
        ];
      }
      ''
        mkdir -p $out
        rsync -a ${drv}/ $out/
        for f in ${drv}/bin/*; do
          cp $f $out/bin
          exe="$out/bin/$(basename $f)"
          chmod +w $exe
          patchelf --remove-rpath $exe
          patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 $exe
        done
        find $out -name '*.so' -exec patchelf --remove-rpath {} \;
      '';

}
