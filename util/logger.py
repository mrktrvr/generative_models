from logging import getLogger
from logging import StreamHandler
from logging import Formatter
from logging import FileHandler
from logging import DEBUG
from logging import INFO
from logging import WARN
from logging import ERROR
from logging import CRITICAL

loggers = {}

level = 'info'

# level = 'debug'

# level = 'warn'


def _set_logger_core(name, level, to_file):
    fmt = '%(asctime)s|%(levelname)s|'
    fmt += '%(filename)s(%(lineno)d) %(funcName)s|%(message)s'
    logger_levels = {
        'debug': DEBUG,
        'info': INFO,
        'warn': WARN,
        'error': ERROR,
        'clitical': CRITICAL
    }
    logger = getLogger(name)
    if logger.handlers:
        logger.handlers = []
    if to_file:
        handler = FileHandler('/tmp/%s.log' % name)
    else:
        handler = StreamHandler()
    formatter = Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logger_levels[level])
    return logger


def set_logger(name=None, level='info', to_file=False):
    global loggers
    name = 'logger' if name is None else name
    logger = loggers.get(name, None)
    if True:
        if logger is not None:
            map(logger.removeHandler, logger.handlers[:])
            map(logger.removeFilter, logger.filters[:])
        logger = _set_logger_core(name, level, to_file)
        loggers.update({name: logger})
    else:
        if logger is None:
            logger = _set_logger_core(name, level, to_file)
            loggers.update({name: logger})
    return logger


logger = set_logger(level=level)


def main():
    logger.debug('debug')
    logger.info('info')
    logger.warn('warn')
    logger.error('error')
    logger.critical('critical')


if __name__ == '__main__':
    main()
