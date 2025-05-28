{
  buildWorkspacePackage,
  readCargo,
}:
buildWorkspacePackage {
  inherit ((readCargo ../../crates/propag/Cargo.toml).package) version;
  pname = "geometry";
  buildAndTestSubdir = "geometry";
}
