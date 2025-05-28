{
  makeWrapper,
  nixosTests,
  symlinkJoin,

  extraPythonPackages ? (ps: [ ]),
  extraPlugins ? [ ],

  libsForQt5,

  qgis,

}:
let
  qgis-unwrapped = qgis;
in
symlinkJoin rec {

  inherit (qgis-unwrapped) version;
  name = "qgis-${version}";

  paths = [ qgis-unwrapped ] ++ extraPlugins;

  nativeBuildInputs = [
    makeWrapper
    qgis-unwrapped.py.pkgs.wrapPython
  ];

  # extend to add to the python environment of QGIS without rebuilding QGIS application.
  pythonInputs = qgis-unwrapped.pythonBuildInputs ++ (extraPythonPackages qgis-unwrapped.py.pkgs);

  postBuild = ''
    buildPythonPath "$pythonInputs"

    for program in $out/bin/*; do
      wrapProgram $program \
        --prefix PATH : $program_PATH \
        --set PYTHONPATH $program_PYTHONPATH
    done
  '';

  passthru = {
    unwrapped = qgis-unwrapped;
    tests.qgis = nixosTests.qgis;
  };

  meta = qgis-unwrapped.meta;
}
