{
  lib,
  stdenv,
  callPackage,
  fetchFromGitHub,
  fetchpatch,

  useMinimalFeatures ? true,
  useArmadillo ? (!useMinimalFeatures),
  useArrow ? (!useMinimalFeatures),
  useHDF ? (!useMinimalFeatures),
  useJava ? (!useMinimalFeatures),
  useLibHEIF ? (!useMinimalFeatures),
  useLibJXL ? (!useMinimalFeatures),
  useMysql ? (!useMinimalFeatures),
  useNetCDF ? (!useMinimalFeatures),
  usePoppler ? (!useMinimalFeatures),
  usePostgres ? (!useMinimalFeatures),
  useTiledb ?
    (!useMinimalFeatures) && !(stdenv.hostPlatform.isDarwin && stdenv.hostPlatform.isx86_64),

  ant,
  armadillo,
  arrow-cpp,
  bison,
  brunsli,
  c-blosc,
  cfitsio,
  cmake,
  crunch,
  cryptopp,
  curl,
  dav1d,
  doxygen,
  expat,
  geos,
  giflib,
  graphviz,
  gtest,
  hdf4,
  hdf5-cpp,
  jdk,
  json_c,
  lerc,
  libaom,
  libde265,
  libdeflate,
  libgeotiff,
  libheif,
  libhwy,
  libiconv,
  libjpeg,
  libjxl,
  libmysqlclient,
  libpng,
  libspatialite,
  libtiff,
  libwebp,
  libxml2,
  lz4,
  netcdf,
  openexr_3,
  openjpeg,
  openssl,
  pcre2,
  pkg-config,
  poppler,
  postgresql,
  proj,
  qhull,
  rav1e,
  sqlite,
  swig,
  tiledb,
  x265,
  xercesc,
  xz,
  zlib,
  zstd,
  ...
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "gdal" + lib.optionalString useMinimalFeatures "-minimal";
  version = "3.8.4"; # same version as ubuntu noble

  src = fetchFromGitHub {
    owner = "OSGeo";
    repo = "gdal";
    rev = "v${finalAttrs.version}";
    hash = "sha256-R9VLof13OXPbWGHOG1Q4WZWSPoF739C6WuNWxoIwKTw=";
  };

  patches = [
    (fetchpatch {
      url = "https://github.com/OSGeo/gdal/commit/40c3212fe4ba93e5176df4cd8ae5e29e06bb6027.patch";
      sha256 = "sha256-D55iT6E/YdpSyfN7KUDTh1gdmIDLHXW4VC5d6D9B7ls=";
    })
    (fetchpatch {
      name = "arrow-18.patch";
      url = "https://github.com/OSGeo/gdal/commit/9a8c5c031404bbc81445291bad128bc13766cafa.patch";
      sha256 = "sha256-tF46DmF7ZReqY8ACTTPXohWLsRn8lVxhKF1s+r254KM=";
    })
  ];

  nativeBuildInputs = [
    bison
    cmake
    doxygen
    graphviz
    pkg-config
  ];

  cmakeFlags = [
    "-DGDAL_USE_INTERNAL_LIBS=OFF"
  ];

  buildInputs = [
    tiledb
    c-blosc
    brunsli
    curl
    libdeflate
    expat
    libgeotiff
    geos
    giflib
    libjpeg
    json_c
    lerc
    xz
    (libxml2.override { enableHttp = true; })
    lz4
    openjpeg
    openssl
    pcre2
    libpng
    proj
    qhull
    libspatialite
    sqlite
    libtiff
    libwebp
    zlib
    zstd
  ];
  enableParallelBuilding = true;
})
