import logging
import time

from tensorboardX import SummaryWriter
from torch import distributed as dist


__all__ = ['Noop', 'TensorBoardWriter', 'init_process_group']


class Noop:
    """A Noop object can be a placeholder for any interface::

       tb = SummaryWriter() if rank == 0 else Noop()

    """

    def __init__(self, *args, **kwargs):
        pass

    def __noop(self, *args, **kwargs):
        return self
    __call__ = __getattr__ = __getitem__ = __noop
    del __noop


class TensorBoardWriter:
    """A wrapper of a TensorBoard-X SummaryWriter. It takes the epoch and the
    iteration instead of global step to encapsulate the global step with more
    familiar variables.

    To write a log at the beginning of an epoch, call it with the epoch and
    chain the logging methods::

       tb = TensorBoardWriter(len(loader), '/var/tmp/tb')
       tb(epoch).scalar('lr', lr)

    It doesn't allow logging at every iteration. Instead, it logs only 1000
    times in an epoch. So many loggings should be skipped. Call it with the
    epoch and the iteration. Then it returns a writer or ``False``. Test the
    return value to determine whether logging or not::

       for i, minibatch in enumerate(loader):
           tb_add = tb(epoch, i)
           if tb_add:
               tb_add.scalar('loss', float(loss))
           ...

    """

    def __init__(self, num_minibatches, path=None, global_steps_per_epoch=1000):
        if path is None:
            self.tb = None
        else:
            self.tb = SummaryWriter(path)

        self.num_minibatches = num_minibatches
        self.global_steps_per_epoch = global_steps_per_epoch
        self.last_global_step = -1

    def __call__(self, epoch=0, i=None):
        if i is None:
            global_step = int(epoch * self.global_steps_per_epoch)

        else:
            epoch_as_float = epoch + i / self.num_minibatches
            global_step = int(epoch_as_float * self.global_steps_per_epoch)

            if self.last_global_step == global_step:
                return False

            self.last_global_step = global_step

        return self.TensorBoardWriterAt(self.tb, global_step)

    class TensorBoardWriterAt:

        def __init__(self, tb, global_step):
            self.tb = tb
            self.global_step = global_step

        def scalar(self, name, value):
            """
            :param name: the name of a plot.
            :param value_fn: the value or a function that returns the value.
            """
            if self.tb is None:
                logging.info('%s: %.5f (%d)', name, value, self.global_step)
            else:
                self.tb.add_scalar(name, value, self.global_step)


def init_process_group():
    while True:
        try:
            dist.init_process_group('nccl')
        except (RuntimeError, ValueError):
            # RuntimeError: Connection timed out
            time.sleep(5)
            continue
        else:
            break

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logging.info('process group: rank-%d among %d processes', rank, world_size)

    return rank, world_size
