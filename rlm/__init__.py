# Auto-load .env file on import
from dotenv import load_dotenv
load_dotenv()  # Loads .env from current working directory

from rlm.core.rlm import RLM

__all__ = ["RLM"]
