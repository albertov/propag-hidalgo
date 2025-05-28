{
  buildWorkspacePackage,
  myRustToolchain,
  cudaPackages,
  openssl,
  gdal,
  removeReferencesTo,
  cudaCombined,
  pkg-config,
  llvmPackages_7,
  ncurses,
  which,
  readCargo,
  proj,
  ubuntize,
}:
let
  self = buildWorkspacePackage {
    inherit ((readCargo ../../crates/propag/Cargo.toml).package) version;
    pname = "propag";
    buildAndTestSubdir = "propag";
    GDAL_DATA = "${gdal}/share/gdal";
    PROJ_DATA = "${proj}/share/proj";
    buildInputs = [
      cudaPackages.cuda_cudart.static
      gdal
      openssl
    ];
    passthru = {
      ubuntuDeps = [
        "libcudart12"
        "libgdal34t64"
      ];
    };
    nativeBuildInputs = [
      cudaCombined
      pkg-config
      llvmPackages_7.llvm
      ncurses # nvmm backend needs it
      which
    ];

    meta.mainProgram = "propag";

    postInstall = ''
      cp -a target/cuda/include $out/
      cp -a target/release/libgeometry.so $out/lib/
    '';
  };
in
self
