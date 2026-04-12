from src.exception.exception import NetworkSecurityException as NetException
from src.logging.logger import logging
from src.utils.utils import read_yaml_file

import os
import sys
from typing import Any
import importlib

def import_class(
        class_path: str
) -> Any:
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        logging.info(f"Class {class_name} imported successfully from module {module_path}")
        return getattr(module, class_name)
    except Exception as e:
        logging.error(f"Error occurred while importing class {class_path}: {e}")
        raise NetException(e, sys)
