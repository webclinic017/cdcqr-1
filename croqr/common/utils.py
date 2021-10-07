import logging
import multiprocessing
import sys
import time
from collections import OrderedDict
from functools import wraps


def timeit(method):
    """
    Decorator used to time the execution time of the downstream function.
    :param method: downstream function
    """

    @wraps(method)
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        if end_time - start_time > 3:
            print('%r  %2.2f sec' % (method.__name__, end_time - start_time))
        return result

    return timed


@timeit
def parallel_jobs(func2apply, domain_list,
                  message='processing individual tasks', num_process=None):
    """
    use multiprocessing module to parallel job
    return a dictionary containing key and corresponding results
    """
    ret_dict = OrderedDict()
    with multiprocessing.Pool(num_process) as pool:
        for idx, ret in enumerate(
                pool.imap_unordered(func2apply, domain_list)):
            ret_dict[domain_list[idx]] = ret
            sys.stderr.write('\r{0} {1:%}'.format(message,
                                                  idx / len(domain_list)))

    return ret_dict


def print_time_from_t0(start_time):
    end_time = time.time()
    print('%2.2f sec' % (end_time - start_time))


def setup_custom_logger(abs_file, log_level=logging.DEBUG):
    """
    Sets up the custom logger with logging formats applied.
    :param abs_file: absolute log file path
    :param log_level: logging level
    :return: logger instance
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # create/open log file
    handler = logging.FileHandler(abs_file)
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(abs_file)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
