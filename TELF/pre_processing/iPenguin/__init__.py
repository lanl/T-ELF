import sys
sys.path += ["OSTI"]
sys.path += ["Scopus"]
sys.path += ["scripts"]
sys.path += ["SemanticScholar"]

from .utils import (multi_urljoin, get_query_param, 
                    get_human_readable_timestamp, format_pubyear)