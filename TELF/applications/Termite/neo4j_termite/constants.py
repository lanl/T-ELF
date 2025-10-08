import numpy as np

"""
General FIELDS
"""
MAKE_ID_UNIQUE = 'unique'

"""
TRIPLET CSV FIELDS
"""
H = 'head'
HT = 'head_type'
T = 'tail'
TT =  'tail_type'
R = 'relation'
W = 'weight'
HA = 'head_attributes'
TA = 'tail_attributes'
ET = "entity_type"
ROW_INDEX = 'index'
FROM_COL = "from_column"
ATTR_COL = 'attribute_columns'
ATTR_FUNC = 'attribute_function'
ATTR_NAME = 'attribute_name'
ARGS = 'args'
RETREIVAL = 'retrival_operation'
ENTITY = 'entity'
ATTRIBUTES = 'attributes'
EXTRACT_H = 'head_extraction_function'
EXTRACT_T = 'tail_extraction_function'
EXTRACT_ENTITY = 'extract_entity'
PAIRING = "ordering_pairing"
HEAD_TO_MANY = 'head_to_many'
INDEX_PAIRING = 'preserve_index'
MANY_TO_MANY = 'many_to_many'
MANY_TO_TAIL = 'many_to_tail'
ARGS_H = 'head_arguments'
ARGS_T = 'tail_arguments'


EMPTY_VALUES = [np.nan, 'None', 'nan']
RETURN_TYPE = {ENTITY:None, W: None, ATTRIBUTES:None}

"""
HEADS AND TAILS / NODE TYPING
"""
YEAR_TYPE = 'Year'
TOPIC_TYPE = 'Topic_ID'
COUNTRY_TYPE = 'Country'
KEYWORD_TYPE = 'Keyword'
AUTHOR_ID_TYPE = 'Author_ID'
DOCUMENT_TYPE = 'Document_ID'
DOCUMENT_TYPE_SCOPUS = 'Document_ID_SCOPUS'
SME_WORDS_TYPE = 'SME_word_tag'
NAMED_ENTITY_TYPE = 'Named_Entity'
AFFILIATION_IDENTIFIER_TYPE = 'Affiliation_ID'
PUBLISHER = 'Publisher'
CATEGORY = 'Scopus_category'

MATERIAL_TYPE = "Material"
AUX_MATERIAL_TYPE = "Auxillary_Material"


NER_LOCATION = "NER_location"
NER_PRODUCT = 'NER_product'
NER_ORGANIZATION = 'NER_organization'
NER_GEOPOLITICAL = 'NER_geopolitical_entity'
SME_KEYWORD = 'sme_keyword'
ACRONYM = 'acronym'



"""
RELATION / EDGE TYPES
"""
DOCUMENT_CITES_RELATION = 'cites'
AUTHOR_DOCUMENT_RELATION = 'wrote'
DOCUMENT_CITED_RELATION = 'cited_by'
TOPIC_KEYWORD_RELATION = 'is_about'
DOCUMENT_YEAR_RELATION = 'written_in_year'
DOCUMENT_TOPIC_RELATION = 'is_part_of_topic'
DOCUMENT_AFFILITATION_RELATION = 'is_affiliated_with'
AUTHOR_AFFILITATION_RELATION = 'is_affiliated_with'
AFFILIATION_COUNTRY_RELATION = 'is_in_country'
DOCUMENT_SME_WORD_RELATION = 'has_sme_word'
DOCUMENT_PUBLISHER_RELATION = 'published_by'
DOCUMENT_CATEGORY_RELATION = 'is_in_category'

DOCUMENT_MATERIAL_RELATION = 'mentions_material'

MATERIAL_TOPIC_RELATION = 'in_topic'
DOCUMENT_MAIN_MATERIAL_RELATION = 'mentions_main_material'
DOCUMENT_AUX_MATERIAL_RELATION = 'mentions_aux_material'

TOPIC_AUX_MATERIAL_RELATION = 'contains_aux_material'
DOCUMENT_SME_RELATION='mentions_sme_keyword'
DOCUMENT_LOCATION_RELATION='mentions_location'
DOCUMENT_PRODUCT_RELATION='mentions_product'
DOCUMENT_ORGANIZATION_RELATION='mentions_organization'
DOCUMENT_GEOPOLITICAL_RELATION='mentions_geopolitical_entity'
DOCUMENT_ACRONYM_RELATION='mentions_acronym'

# WITH ['Topic_ID', 'Keyword', 'Document_ID', 'Document_ID_SCOPUS', 'Affiliation_ID', 'Country',
#       'Year', 'Author_ID', 'SME_word_tag', 'Publisher', 'Scopus_category' ] AS labels
#  FOREACH (label IN labels |
#   CREATE CONSTRAINT FOR (node:`$label`) REQUIRE node.neo4jImportId IS UNIQUE;
# )

