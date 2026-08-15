import logging
import os
import datetime as dt

LOG_FILE_NAME = f"log_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH =os.path.join(log_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started.")




