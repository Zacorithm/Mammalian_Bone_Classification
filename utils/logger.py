import logging
import torch.distributed as dist

logger_initialized = {}

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """get_logger(name) plus a filter that keeps only records whose text contains `name`."""
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Cached logger by name. StreamHandler always; FileHandler only on rank 0; non-zero ranks set to ERROR."""
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        if isinstance(handler, logging.FileHandler):
            # DEBUG-level records (e.g. the full args/config dump via
            # capped at `log_level` so it stays readable
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        # DEBUG so the logger itself never drops a record before it reaches
        # the handlers above — each handler does its own filtering
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger

def print_log(msg, logger=None, level=logging.INFO):
    """logger: a Logger, "silent" (drop), a name for get_root_logger(name), or None (plain print)."""
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')