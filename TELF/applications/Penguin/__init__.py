from .utils import (try_int, get_from_dict,
                    transform_dict, verify_n_jobs,
                    verify_attributes, gen_chunks,
                    match_lists, drop_duplicates,
                    reorder_and_add_columns, merge_frames,
                    match_frames_text, create_text_query,
                    create_general_query)
from .crocodile import (process_scopus_json, process_scopus_xml,
                        process_s2_json, parse_scopus_funding,
                        parse_scopus_affiliations, form_scopus_df,
                        form_s2_df, form_df)
from .penguin import Penguin