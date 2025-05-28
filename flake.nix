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
    devour-flake = {
      url = "github:srid/devour-flake";
      flake = false;
    };
    pre-commit-nix.url = "github:cachix/pre-commit-hooks.nix";
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
        toBareDEB =
          let
            pkgs = self.legacyPackages.x86_64-linux;
          in
          drv:
          let
            deps = drv.ubuntuDeps or [ ];
            depends = pkgs.lib.concatMapStringsSep " " (x: "--deb-pre-depends ${x}") deps;
            pkg = pkgs.ubuntize drv;
          in
          pkgs.runCommand "toBareDEB"
            {
              nativeBuildInputs = with pkgs; [
                fpm
                binutils
                rsync
              ];
            }
            ''
              mkdir -p tmp
              rsync -a ${pkg}/ tmp/
              chmod -R u+rwx tmp/
              pushd tmp
              ${drv.ubuntuPrePackage or ""}
              fpm -s dir -t deb \
                --name ${drv.pname} \
                -v ${drv.version or "0.1"} \
                ${depends} \
                ./
              popd
              mkdir -p $out
              cp -r tmp/*.deb $out
            '';
        toDockerImage =
          { ... }@drv:
          let
            pkgs = nixpkgs.legacyPackages.x86_64-linux;
          in
          pkgs.dockerTools.buildImage {
            name = drv.pname or "image";
            tag = drv.version;
            fromImageName = "ubuntu";
            fromImageTag = "noble";
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
              (import ./nix/overlay.nix self inputs system)
            ];
            config.allowUnfree = true;
          };
          treefmtEval = treefmt-nix.lib.evalModule pkgs ./nix/treefmt.nix;
        in
        {
          legacyPackages = pkgs;
          packages = {
            default = self'.packages.propag;
            propag = pkgs.propag;
            py-propag = pkgs.py-propag;
          };
          apps.make_deb = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "make_deb";
              runtimeInputs = [ pkgs.nix ];
              text = ''
                nix bundle --bundler .#toBareDEB .# -o bundle
                echo Produced DEB package for Ubuntu noble at bundle/*
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
          apps.pushAll = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "nix-push-all";
              runtimeInputs = with pkgs; [
                nix
                attic-client
              ];
              text = ''
                # Make sure that flake.lock is sync
                nix flake lock --no-update-lock-file

                # Do a full nix build (all outputs)
                # This uses https://github.com/srid/devour-flake
                for f in $(nix run .#buildAll -- "$@"); do
                  (nix-store -qR --include-outputs "$(nix-store -qd "$f")" \
                    | grep -v '\.drv$' \
                    | grep -v 'unknown-deriver$' \
                    | xargs -n1000 attic push propag25) || true
                done
              '';
            };
          };
          apps.buildAll = flake-utils.lib.mkApp {
            drv = pkgs.writeShellApplication {
              name = "nix-build-all";
              runtimeInputs = [
                pkgs.nix
                (pkgs.callPackage inputs.devour-flake { })
              ];
              text = ''
                # Make sure that flake.lock is sync
                nix flake lock --no-update-lock-file

                # Do a full nix build (all outputs)
                # This uses https://github.com/srid/devour-flake
                devour-flake . "$@"
              '';
            };
          };

          # Development shell
          devShells.default = import ./nix/devshell.nix pkgs;

          # for `nix fmt`
          formatter = treefmtEval.config.build.wrapper;
          # for `nix flake check`
          checks = {
            formatting = treefmtEval.config.build.check self;
          };
        };
    };
}
