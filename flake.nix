{
  description = "WildFire Propagator 2025";
  nixConfig = {
    bash-prompt = "\\[\\e[0;37m\\](\\[\\e[0m\\]nix) \\[\\e[0;1;91m\\]propag25 \\[\\e[0m\\]\\w \\[\\e[0;1m\\]$ \\[\\e[0m\\]";
    extra-substituters = [
      "https://nixcache.toscat.net/propag25"
    ];
    extra-trusted-public-keys = [
      "propag25:gOQfcU+oofw7ILwjHXQZC6JTlcrc/wrKFn6k7eBN5y0="
    ];
  };
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
    nix-utils = {
      url = "github:albertov/nix-utils";
    };
  };

  outputs =
    inputs@{
      self,
      flake-utils,
      flake-parts,
      treefmt-nix,
      rust-overlay,
      nix-utils,
      nixpkgs,
      ...
    }:
    {
      bundlers.x86_64-linux = {
        toDEB =
          drv:
          nix-utils.bundlers.deb {
            system = "x86_64-linux";
            program = nixpkgs.lib.getExe drv;
            inherit (drv) version;
          };
        toDockerImage =
          { ... }@drv:
          let
            pkgs = nixpkgs.legacyPackages.x86_64-linux;
          in
          pkgs.dockerTools.buildImage {
            name = drv.pname or "image";
            tag = drv.version;
            fromImageName = "nvidia/cuda";
            fromImageTag = "12.6.3-runtime-ubuntu24.04";
            copyToRoot = pkgs.buildEnv {
              name = "image-root";
              paths = [ drv ];
            };
            config.EntryPoint = [ (pkgs.lib.getExe drv) ];
            config.Env = [ "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64" ];
          };
      };
    }
    // flake-parts.lib.mkFlake { inherit inputs; } {
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
              (import ./nix/overlay.nix inputs)
            ];
            config.allowUnfree = true;
          };
          treefmtEval = treefmt-nix.lib.evalModule pkgs ./nix/treefmt.nix;
        in
        {
          legacyPackages = pkgs;
          packages = {
            default = self'.packages.propag;
            inherit (pkgs)
              propag
              ;
          };
          apps.make_deb = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "make_deb";
              runtimeInputs = [ pkgs.nix ];
              text = ''
                nix bundle --bundler .#toDEB .# -o bundle
                echo Produced DEB package at bundle/*
              '';
            };
          };
          apps.make_docker = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "make_docker";
              runtimeInputs = [ pkgs.nix ];
              text = ''
                nix bundle --bundler .#toDockerImage .# -o propag.docker
                echo Produced Docker image at propag.docker
                echo 'Load it with "docker load < propag.docker"'
              '';
            };
          };

          # Development shell
          devShells.default = import ./nix/devshell.nix pkgs;

          # for `nix fmt`
          formatter = treefmtEval.config.build.wrapper;
          # for `nix flake check`
          checks = {
            formatting = treefmtEval.config.build.check self';
          };
        };
    };
}
