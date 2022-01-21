# jax-gpu-cluster-demo
Little demo of how to instantiate a cluster of GPUs with JAX


For context, over the course of 2020-2021, low level NCCL collectives have been implemented directly within the XLA backend in tensorflow. That means that JAX also can now in principle use these collectives for multi-host distribution on GPU clusters! Unfortunately, for now accessing the API to create such a GPU cluster it is still a bit hacky and undocumented, which is the point of this demo.

This little demo, mostly stolen from Trax illustrates how to launch an SPMD jax code on multiple nodes and have all the GPUs visible.

## How does it work

Essentially, we need to access the C++ XLA library through the xla_bridge to start xla server/clients, and then we need to tell Jax to use this distributed xla backend as the backend to use for executing operations
```python
import jax
from jax.lib import xla_extension as xc
...
# Creating a distributed runtime:
# On the server:
service = xc.get_distributed_runtime_service(addr, n_hosts)
# On the client:
dist_client = xc.get_distributed_runtime_client(addr, host_id)
dist_client.connect()
...

# Registering the distributed runtime for use in Jax
  factory = functools.partial(jax.lib.xla_client.make_gpu_client,
                              dist_client, host_id)
  jax._src.lib.xla_bridge.register_backend_factory('gpu', factory, priority=300)
```

## Installing environment on Perlmutter

To install all you need:
```bash
$ module load tensorflow
$ pip install --upgrade --user pip
$ pip install --upgrade --user "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html 
```

## Submitting to SLURM

The slum job provided should allocate a total of 16 GPUs accross 4 nodes:
```bash
$ sbatch slurm_job.sh
```
Of course adjusting as necessary the account number

and this should return a log like this:
```
...
I am 15 I see 1/16 devices
I am 7 I see 1/16 devices
I am 11 I see 1/16 devices
I am 13 I see 1/16 devices
...
```

which shows that we have 16  processes running, each one in charge of 1 GPU \o/ but they can all see all GPUs!
