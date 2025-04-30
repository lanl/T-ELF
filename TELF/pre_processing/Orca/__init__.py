import sys
sys.path += ["DuplicateAuthorFinder"]
sys.path += ["AuthorMatcher"]
sys.path += ["AuthorFrames"]

from .orca import Orca
from .utils import match_name, generate_name_variations