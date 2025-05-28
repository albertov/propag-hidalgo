{
  description = "WildFire Propagator 2025";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
    nixpkgs_old.url = "github:NixOS/nixpkgs/release-23.11";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      treefmt-nix,
      rust-overlay,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        # systems for which you want to build the `perSystem` attributes
        "x86_64-linux"
        "aarch64-darwin"
        # ...
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          system,
          ...
        }:
        let
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              rust-overlay.overlays.default
              (import ./overlay.nix inputs)
              /*
                (_:_: {
                  pkgs_2311 = import inputs.nixpkgs_old {
                    inherit system;
                  };
                })
              */
            ];
            config.allowUnfree = true;
          };
          treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
        in
        {
          legacyPackages = pkgs;
          packages = {
            default = self'.packages.propag;
            inherit (pkgs)
              propag
              firelib
              ;
          };
          devShells.default = pkgs.mkShell {
            inputsFrom = [
              pkgs.firelib
              pkgs.propag
            ];
            env = {
              # FIXME
              LD_LIBRARY_PATH = "${pkgs.linuxPackages.nvidia_x11}/lib";

              GDAL_DATA = "${pkgs.gdal}/share/gdal";
              PROJ_DATA = "${pkgs.proj}/share/proj";

              inherit (pkgs.firelib)
                BINDGEN_EXTRA_CLANG_ARGS
                LIBCLANG_PATH
                ;
              inherit (pkgs.propag)
                CUDA_PATH
                CUDA_ROOT
                LLVM_CONFIG
                LLVM_LINK_SHARED
                ;
            };
            packages = with pkgs; [
              git
              cudaPackages.cuda_cudart
              cargo-watch
              cargo-valgrind
              valgrind
              rustup
              gdb
              (enableDebugging gdal)
            ];
          };
          # for `nix fmt`
          formatter = treefmtEval.config.build.wrapper;
          # for `nix flake check`
          checks = {
            formatting = treefmtEval.config.build.check self';
          };
        };
    };
}
