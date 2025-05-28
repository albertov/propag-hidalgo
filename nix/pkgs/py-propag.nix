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
  passthru = {
    ubuntuDeps = [
      "libcudart12"
      "libgdal34t64"
      "python3"
    ];
    ubuntuPrePackage = ''
      mkdir -p usr/lib/python3/dist-packages/
      mv lib/python3.12/site-packages/* usr/lib/python3/dist-packages/
    '';
  };
}
