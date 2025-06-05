#!/usr/bin/env bash

set -eu -o pipefail

cd crates
nix fmt
cargo clippy --fix --allow-dirty
