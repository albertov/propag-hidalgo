{
  description = "WildFire Propagator 2025";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-24.11";
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
          treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
          pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              rust-overlay.overlays.default
              (import ./overlay.nix)
            ];
            config.allowUnfree = true;
          };
        in
        {
          packages = {
            default = self'.packages.firelib-rs;
            inherit (pkgs) firelib-rs;
          };
          devShells.default = pkgs.mkShell {
            /*
              shellHook = with pkgs; ''
                  export BINDGEN_EXTRA_CLANG_ARGS="$(< ${stdenv.cc}/nix-support/libc-crt1-cflags) \
                  $(< ${stdenv.cc}/nix-support/libc-cflags) \
                  $(< ${stdenv.cc}/nix-support/cc-cflags) \
                  $(< ${stdenv.cc}/nix-support/libcxx-cxxflags) \
                  ${lib.optionalString stdenv.cc.isClang "-idirafter ${stdenv.cc.cc}/lib/clang/${lib.getVersion stdenv.cc.cc}/include"} \
                  ${lib.optionalString stdenv.cc.isGNU "-isystem
                  ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}
                  -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config} -isystem ${buildPackages.llvmPackages.libclang.lib}/lib/clang/${builtins.head (lib.splitString "." (lib.getVersion buildPackages.clang))}/include"}
                  "
                export LIBCLANG_PATH="${buildPackages.llvmPackages.libclang.lib}/lib"
              '';
            */
            # ${lib.optionalString stdenv.cc.isGNU "-isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc} -isystem ${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"}
            inputsFrom = [
              self'.packages.firelib-rs
            ];
            packages = with pkgs; [
              git
              cargo
              cargo-watch
              rust-bindgen
              rustfmt
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
