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
}:
buildWorkspacePackage {
  inherit ((readCargo ../../crates/propag/Cargo.toml).package) version;
  pname = "propag";
  buildAndTestSubdir = "propag";
  GDAL_DATA = "${gdal}/share/gdal";
  PROJ_DATA = "${proj}/share/proj";
  buildInputs = [
    cudaPackages.cuda_cudart.static
    openssl
    gdal
  ];
  nativeBuildInputs = [
    cudaCombined
    pkg-config
    llvmPackages_7.llvm
    ncurses # nvmm backend needs it
    which
  ];

  meta.mainProgram = "propag";

}
