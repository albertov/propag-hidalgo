pkgs:
pkgs.mkShell {
  inputsFrom = [
    pkgs.propag
  ];
  shellHook = ''
    ${pkgs.pre-commit.shellHook}
  '';
  env = {
    # So we can link against CUDA runtime API while developing
    LD_LIBRARY_PATH = "${pkgs.linuxPackages.nvidia_x11}/lib";

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
    linuxPackages.nvidia_x11
    linuxPackages.nvidia_x11.bin
    git
    cudaPackages.cuda_gdb
    cargo-watch
    cargo-valgrind
    valgrind
    rustup
    gdb
    python3
    python3.pkgs.ipython
    python3.pkgs.gdal
    python3.pkgs.numpy
    maturin
    gdal
  ];
}
