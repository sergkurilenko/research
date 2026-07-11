# Same-hardware Linux/TenSEAL point

This image is intentionally small: it supplies Python 3.11, NumPy 2.4.3, and
TenSEAL 0.3.16, then runs the repository code from a read/write bind mount.
All native thread-pool variables are fixed to one.

Build from `revision_workspace`:

```powershell
docker build `
  --file system\docker\Dockerfile.tenseal-linux `
  --tag kurilenko-tenseal-linux:py311 `
  system\docker
```

Run the exact packed-CKKS microbenchmark:

```powershell
docker run --rm `
  --mount "type=bind,source=$PWD,target=/work" `
  kurilenko-tenseal-linux:py311 `
  system/packed_ckks.py `
  --dimension 672 --candidates 100 --repeats 20 --warmups 2 `
  --seed 20260710 `
  --output results/system_revision/systems_expansion/linux_docker_ckks_micro.json
```

Docker Desktop uses its WSL2 Linux VM on the same physical i5-14400F host.
Consequently this is a same-hardware, different-OS/runtime portability point,
not an independent-machine replication and not native bare-metal Linux.
