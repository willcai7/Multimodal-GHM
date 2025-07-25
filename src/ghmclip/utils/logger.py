import logging 
import os 
from dataclasses import asdict
import json

def GenLogger(directory,config, raw=True):
    os.makedirs(directory, exist_ok=True)
    config_file_path = os.path.join(directory, 'config.json') # path to save the configuration file
    log_file_path = os.path.join(directory, 'training.log') # path to save the log file    
    logger = logging.getLogger() # logger object
    logger.setLevel(logging.DEBUG) # set the logging level
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # format of the log message
    console_handler = logging.StreamHandler() # console handler
    console_handler.setLevel(logging.INFO) # set the logging level
    console_handler.setFormatter(formatter) # set the formatter
    logger.addHandler(console_handler) # add the console handler to the logger

    if not raw:
        json.dump(asdict(config), open(config_file_path, 'w'), indent=4) # save the configuration
        file_handler2 = logging.FileHandler(log_file_path, mode="a") # file handler
        file_handler2.setLevel(logging.DEBUG) # set the logging level
        file_handler2.setFormatter(formatter) # set the formatter

        logger.addHandler(console_handler) # add the console handler to the logger
        logger.addHandler(file_handler2) # add the file handler to the logger
        logger.info(f'Training with config: {asdict(config)}') # log the configuration
    return logger # return the logger object

