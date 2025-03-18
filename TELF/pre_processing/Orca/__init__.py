import sys
sys.path += ["DuplicateAuthorFinder"]
sys.path += ["AuthorMatcher"]
sys.path += ["AuthorFrames"]

from .orca import Orca
from .utils import verify_n_jobs, get_from_dict, match_name, generate_name_variations