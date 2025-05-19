import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

def info(s: str):
    logging.info(s)

def warn(s: str):
    logging.warning(s)

def error(s: str):
    logging.error(s)

