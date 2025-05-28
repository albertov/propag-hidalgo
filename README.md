# propag25


# Install (Linux)

There are two ways to build this project, with or without Nix. Nix is
recommended for CUDA development since the dependencies for it are installed
automatically.

## Without Nix (eg Ubuntu)

### Install rustup

Rustup takes care of installing the Rust compiler (rustc), the rust package manager
(Cargo) and other tools

```console
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s
```

### Build

```console
cargo build
```

### Run tests

```console
cargo test
```
