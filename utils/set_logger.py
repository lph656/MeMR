import logging

def set_logger(logfile):
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    # Rebuild handlers so each run starts with a clean logger configuration.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    fh = logging.FileHandler(filename=logfile, encoding="utf-8", mode="w")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
