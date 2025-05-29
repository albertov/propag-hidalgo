# propag25

[![.github/workflows/test.yml](https://github.com/albertov/propag25/actions/workflows/test.yml/badge.svg)](https://github.com/albertov/propag25/actions/workflows/test.yml)

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

### Running it on Meluxina

```console
(login)$ salloc -A p200648 -t 01:00:00 -q dev --res gpudev -p gpu -N 1 --ntasks-per-node=4
(compute)$ module load Apptainer/1.3.6-GCCcore-13.3.0
(compute)$ module load OpenMPI/5.0.3-NVHPC-24.9-CUDA-12.6.0
(compute)$ export IMAGE=albertometeo/propag:0.1.0
(compute)$ apptainer pull docker://$IMAGE
(compute)$ mpirun -n 4 -bind-to none -map-by :OVERSUBSCRIBE -- apptainer run --nv propag_0.1.0.sif
```

# Sponsors

This application has been developed within the frame and for the purpose of the
HiDALGO2 project, funded by the European Union. This work has received funding
from the European High Performance Computing Joint Undertaking (JU) and Poland,
Germany, Spain, Hungary, France, Greece under grant agreement number: 101093457.

[![HiDALGO2](https://www.hidalgo2.eu/wp-content/uploads/2023/07/cropped-cropped-HiDALGO2-logo-color-RGB-768x212.png)](https://www.hidalgo2.eu)
[![Funded by the European Union](./EN_FundedbytheEU_RGB_POS.png)](https://ec.europa.eu)
