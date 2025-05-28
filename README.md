# propag25


# Development (Linux)

## Install nix

TBD

## Build

Enter the development shell

```console
$ nix develop
```

Inside the dev shell run cargo

```console
$ cd crates
$ cargo build
```

## Run tests

```console
$ cargo test
```

# Building a Debian package

```console
$ nix run .#make_deb
```

This will produce a Debian package inside a directory symlinked at `bundle`

```console
$ ls bundle/propag-git-bin-propag_1.0_amd64.deb 
bundle/propag-git-bin-propag_1.0_amd64.deb
```

# Testing the Debian package

After building the package, start up a docker container. The NVIDA Container
Toolkit installed needs to be installed on the host.

```console
$ export IMAGE="nvidia/cuda:12.6.3-runtime-ubuntu24.04"
$ docker run --device nvidia.com/gpu=all -it --rm -v $(readlink -f bundle):/bundle $IMAGE
```

Inside the container, install the `*.deb` package in `/package` and run `propag`

```console
# apt update
(....)
# apt install /bundle/*.deb
(....)
# propag
Calculating with GPU Propag
fire_pos=USizeVec2(500, 500)
Generating input data
Loading module
Getting function
max_active_blocks_per_multiprocessor=1
max_total_blocks=128
using geo_ref=GeoReference { width: 1024, height: 1024, epsg: 25830, transform:
GeoTransform { gt: GT { x0: 0.0, dx: 5.0, rx: 0.0, y0: 0.0, ry: 0.0, dy: 5.0 },
inv: GT { x0: 0.0, dx: 0.19999999, rx: -0.0, y0: 0.0, ry: -0.0, dy: 0.19999999 }
} }
grid_size=GridSize { x: 11, y: 11, z: 1 }
blocks_size=BlockSize { x: 24, y: 24, z: 1 }
super_grid_size=(4, 4)
for 1048576 elems
10 loops: 79.4600234 ms
config_max_time=36000
max_time=Some(35999.9)
max_time=Some(0.0)
num_times_after=447822
Generating times geotiff
```

(or just install it without docker if on a debian-based system)

## Building a Docker image

```console
$ nix run .#make_docker
Produced Docker image at propag.docker
Load it with "docker load < propag.docker"
$ docker load < propag.docker
a14bc4312f8b: Loading layer
[==================================================>]  190.4MB/190.4MB
Loaded image: propag:0.1.0  
```

### Running it

To run the docker image the NVIDA Container Toolkit installed needs to be
installed on the host.

```console
$ docker run --device nvidia.com/gpu=all -it --rm propag:0.1.0
Calculating with GPU Propag
fire_pos=USizeVec2(500, 500)
Generating input data
Loading module
Getting function
max_active_blocks_per_multiprocessor=1
max_total_blocks=128
using geo_ref=GeoReference { width: 1024, height: 1024, epsg: 25830, transform: GeoTransform { gt: GT { x0: 0.0, dx: 5.0, rx: 0.0, y0: 0.0, ry: 0.0, dy: 5.0 }, inv: GT { x0: 0.0, dx: 0.19999999, rx: -0.0, y0: 0.0, ry: -0.0, dy: 0.19999999 } } }
grid_size=GridSize { x: 11, y: 11, z: 1 }
blocks_size=BlockSize { x: 24, y: 24, z: 1 }
super_grid_size=(4, 4)
for 1048576 elems
10 loops: 79.4684815 ms
config_max_time=36000
max_time=Some(35999.9)
max_time=Some(0.0)
num_times_after=447822
Generating times geotiff
```
