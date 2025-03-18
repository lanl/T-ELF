import sys
sys.path += ["OSTI"]
sys.path += ["Scopus"]
sys.path += ["scripts"]
sys.path += ["SemanticScholar"]

from .utils import (multi_urljoin, get_query_param, 
                    get_human_readable_timestamp, format_pubyear, 
                    chunk_dict, get_from_dict, 
                    gen_chunks, try_int)