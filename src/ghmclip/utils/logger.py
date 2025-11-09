import logging 
import os 
from dataclasses import asdict

def GenLogger(directory,config, raw=True):
    config_file_path = os.path.join(directory, 'config.log') # path to save the configuration file
    log_file_path = os.path.join(directory, 'training.log') # path to save the log file    
    logger = logging.getLogger() # logger object
    logger.setLevel(logging.DEBUG) # set the logging level
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # format of the log message
    console_handler = logging.StreamHandler() # console handler
    console_handler.setLevel(logging.INFO) # set the logging level
    console_handler.setFormatter(formatter) # set the formatter
    logger.addHandler(console_handler) # add the console handler to the logger

    if not raw:
        os.makedirs(directory, exist_ok=True) # create the directory
        file_handler1 = logging.FileHandler(config_file_path, mode="a") # file handler
        file_handler1.setLevel(logging.DEBUG) # set the logging level
        file_handler1.setFormatter(formatter) # set the formatter

        file_handler2 = logging.FileHandler(log_file_path, mode="a") # file handler
        file_handler2.setLevel(logging.DEBUG) # set the logging level
        file_handler2.setFormatter(formatter) # set the formatter

        logger.addHandler(console_handler) # add the console handler to the logger
        logger.addHandler(file_handler1) # add the file handler to the logger
        logger.info(f'Training with config: {asdict(config)}') # log the configuration
        logger.removeHandler(file_handler1) # remove the file handler 
        file_handler1.close() # close the file handler
        logger.addHandler(file_handler2) # add the file handler to the logger
    return logger # return the logger object

