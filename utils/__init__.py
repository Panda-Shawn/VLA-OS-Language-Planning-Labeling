from .cam_utils import *
from .utils import *
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"

PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompts"