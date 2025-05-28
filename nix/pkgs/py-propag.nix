{
  buildWorkspacePythonPackage,
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
buildWorkspacePythonPackage {
  inherit ((readCargo ../../crates/py-propag/Cargo.toml).package) version;
  pname = "py-propag";
  buildAndTestSubdir = "py-propag";
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
}
