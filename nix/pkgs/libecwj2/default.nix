{
  stdenv,
  lib,
  fetchurl,
  zlib,
  libtiff,
  libjpeg,
}:

stdenv.mkDerivation rec {
  version = "3.3";
  shortname = "libecwj2";
  name = "${shortname}-${version}";

  src = fetchurl {
    url = "https://github.com/albertov/${shortname}/archive/${version}.tar.gz";
    sha256 = "1fx0pgasz2z6h5vj0l41900x595xn3b92qkxnnjipyhpa3dzh8ac";
  };

  # fixme!
  hardeningDisable = [ "format" ];

  enableParallelBuilding = true;

  buildInputs = [
    zlib
    libtiff
    libjpeg
  ];

  # prevent linking impure paths
  preConfigure = ''
    sed -i 's:-R/usr/local/lib::g' configure
  '';

  preInstall = "mkdir -p $out/include";

  patches = [ ./libecwj2-3.3.patch ];

  meta = {
    description = "Library for reading ECW files";
    homepage = "http://www.hexagongeospatial.com/";
    license = lib.licenses.gpl2;
  };
}
