from datetime import datetime
import uvicorn
from fastapi import FastAPI
import logging
from pathlib import Path
from api.routers.general import general
from api.routers.inference import inference
from api.config import config

# create results directory
time_str = str(datetime.now()).replace(" ", "_")
log_file = Path(config.logging.output_dir).joinpath(f"output_{time_str}.log")
if not log_file.parent.exists():
    log_file.parent.mkdir(parents=True)

# setup logging for file and stderr/stdout
root_logger = logging.getLogger()
root_logger.setLevel(logging._nameToLevel[str(config.logging.level).upper()])
file_handler = logging.FileHandler(str(log_file))
file_handler.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging._nameToLevel[str(config.logging.level).upper()])
root_logger.addHandler(console_handler)

app = FastAPI(title=config.title,
              description=config.description,
              version=config.version)

app.include_router(general)
app.include_router(inference, prefix="/inference")

if __name__ == "__main__":
    # start the api via uvicorn
    assert config.port is not None and isinstance(config.port, int), "The port has to be an integer! E.g. 8080"
    uvicorn.run(app,
                host="0.0.0.0",
                port=config.port,
                debug=config.debug)
