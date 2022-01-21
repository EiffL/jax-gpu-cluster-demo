## This code is stollen from https://github.com/google/trax/blob/master/trax/trainer.py
from absl import app
from absl import flags
from absl import logging

import functools
import atexit

import jax
from jax.lib import xla_extension as xc

FLAGS = flags.FLAGS

# JAX/XLA GPU cluster flags.
flags.DEFINE_string('gpu_cluster_chief_ip', '0.0.0.0', 'IP of GPU cluster chief.')
flags.DEFINE_integer('gpu_cluster_n_hosts', 1,
                     'Number of hosts in GPU cluster.')
flags.DEFINE_integer('gpu_cluster_host_id', 0, 'Host id inside GPU cluster.')
flags.DEFINE_integer('gpu_cluster_port', 5005, 'Port to use in GPU cluster.')

flags.DEFINE_integer('log_level', logging.INFO, 'Log level.')

def make_jax_gpu_cluster(host_id, server_ip, n_hosts, server_port=5005):
  """Make JAX GPU Cluster."""

  addr = f'{server_ip}:{server_port}'
  if host_id == 0:
    logging.info('starting service on %s', addr)
    service = xc.get_distributed_runtime_service(addr, n_hosts)
    # We add an explicit call to shutdown the service via atexit as Python
    # interpreter may not call the service destructor on process termination.
    atexit.register(service.shutdown)

  logging.info('connecting to service on %s', addr)
  dist_client = xc.get_distributed_runtime_client(addr, host_id)
  dist_client.connect()
  atexit.register(dist_client.shutdown)

  # register dist gpu backend
  factory = functools.partial(jax.lib.xla_client.make_gpu_client,
                              dist_client, host_id)
  jax._src.lib.xla_bridge.register_backend_factory('gpu', factory, priority=300)



def main(_):
  logging.set_verbosity(FLAGS.log_level)

  # Initialize the cluster
  make_jax_gpu_cluster(FLAGS.gpu_cluster_host_id,
                       FLAGS.gpu_cluster_chief_ip,
                       FLAGS.gpu_cluster_n_hosts,
                       FLAGS.gpu_cluster_port)

  print('I am %d'%FLAGS.gpu_cluster_host_id, 'I see %d/%d devices'%(jax.local_device_count(), jax.device_count()))
    
    
if __name__ == '__main__':
  app.run(main)