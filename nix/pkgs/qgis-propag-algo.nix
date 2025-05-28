{
  lib,
  cmake,
  stdenv,
  qgis,
  qt5,
}:
let
  self = stdenv.mkDerivation {
    name = "qgis-propag-algo";
    src = lib.cleanSource ../../qgis-plugin/propag-algo;
    pname = "qgis-propag-algo";
    nativeBuildInputs = [ cmake ];
    cmakeFlags = [ "-DQGIS_PREFIX_PATH=${qgis}" ];
    buildInputs = [
      qt5.full
    ];
  };
in
self
