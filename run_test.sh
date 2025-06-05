#!/usr/bin/env bash
cd crates
exec cargo test "$@"
