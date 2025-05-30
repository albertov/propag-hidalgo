pkgs:
let
  qgis = pkgs.qgis-orig;
in
pkgs.mkShell {
  inputsFrom = [
    pkgs.propag
    #pkgs.py-propag
    #pkgs.qgis-propag-algo
  ];
  shellHook = ''
    ${pkgs.pre-commit.shellHook}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/crates/target/debug
  '';
  env = {
    # So we can link against CUDA runtime API while developing
    LD_LIBRARY_PATH = "/run/opengl-driver/lib/";

    QGIS_PREFIX_PATH = qgis;

    inherit (pkgs.propag)
      GDAL_DATA
      PROJ_DATA
      LIBCLANG_PATH
      CUDA_PATH
      CUDA_ROOT
      LLVM_CONFIG
      LLVM_LINK_SHARED
      ;
  };
  packages = with pkgs; [
    cmake
    git
    cudaPackages.cuda_gdb
    cargo-watch
    cargo-valgrind
    valgrind
    rustup
    gdb
    qgis
    python3
    python3.pkgs.ipython
    python3.pkgs.gdal
    python3.pkgs.numpy
    python3.pkgs.pyqt5
    python3.pkgs.pyqt5
    python3.pkgs.pyqt-builder
    maturin
    gdal
    qtcreator
    qt5.full
    #clippy
  ];
}
