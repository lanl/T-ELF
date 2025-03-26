from .parser import ScopusQueryParser
from .scopusAPI import ScopusAPI
from .scopus import Scopus
from .process_scopus_json import (gen_chunks, parse_funding,
                                  match_lists, parse_affiliation,
                                  form_df)