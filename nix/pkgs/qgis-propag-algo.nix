{
  lib,
  cmake,
  stdenv,
  qgis,
  qt5,
  propag,
  geometry,
}:
let
  self = stdenv.mkDerivation {
    name = "qgis-propag-algo";
    src = lib.cleanSource ../../qgis-plugin/propag-algo;
    pname = "qgis-propag-algo";
    nativeBuildInputs = [ cmake ];
    cmakeFlags = [
      "-DQGIS_PREFIX_PATH=${qgis}"
    ];
    buildInputs = [
      qgis
      propag
      geometry
      qt5.full
    ];
    postFixup = ''
      patchelf --add-rpath ${propag}/lib $out/lib/qgis/plugins/libpropagplugin.so
    '';
  };
in
self
