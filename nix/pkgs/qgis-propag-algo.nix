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
      "-DPROPAG_PREFIX_PATH=${propag}"
      "-DGEOMETRY_PREFIX_PATH=${geometry}"
    ];
    buildInputs = [
      qgis
      propag
      geometry
      qt5.full
    ];
  };
in
self
