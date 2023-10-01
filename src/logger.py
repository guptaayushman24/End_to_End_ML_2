import sys
import logging
import os
from datetime import datetime
sys.path.append('C:\\Users\\gupta\\OneDrive\\Desktop\\End to End ML Project-2')

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s %(lineno)d %(name)s-%(levelname)s-%(message)s]",
    level=logging.INFO
)

# After creating the logging file we need to check logging is working or not
if __name__ == '__main__' :
    logging.info("Logging has started")


