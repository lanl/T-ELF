from .utils import (create_text_query, create_general_query)
from .crocodile import (process_scopus_json, process_scopus_xml,
                        process_s2_json, parse_scopus_funding,
                        parse_scopus_affiliations, form_scopus_df,
                        form_s2_df, form_df)
from .penguin import Penguin