import logging
import os
import torch.distributed as dist


def create_logger(logging_dir, logger_name=__name__, use_color=True):
    """
    Create a logger that writes to a log file and stdout.

    Args:
        logging_dir (str): Directory where the log file will be saved.
        logger_name (str): Name of the logger.
        use_color (bool): Whether to use ANSI colors in the console output.

    Returns:
        logging.Logger: Configured logger.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized.")

    rank = dist.get_rank()
    logger = logging.getLogger(logger_name)
    
    if not logger.handlers:
        if rank == 0:  
            logger.setLevel(logging.INFO)

            os.makedirs(logging_dir, exist_ok=True)
            log_file = os.path.join(logging_dir, "log.txt")

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            if use_color:
                console_formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_formatter = ColoredFormatter('[%(asctime)s] %(levelname)s - %(message)s',
                                                     datefmt='%Y-%m-%d %H:%M:%S')
            else:
                console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s',
                                                      datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(console_formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        else:
            logger.setLevel(logging.WARNING)
            logger.addHandler(logging.NullHandler())

    return logger

class ColoredFormatter(logging.Formatter):
    """
    Custom colored formatter
    """
    COLOR_MAP = {
        'DEBUG': '\033[37m',    # white
        'INFO': '\033[32m',     # green
        'WARNING': '\033[33m',  # yellow
        'ERROR': '\033[31m',    # red
        'CRITICAL': '\033[35m', # purple
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

from time import strftime

def print_with_prefix(*messages):
    prefix = f"\033[34m[LightningDiT-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

