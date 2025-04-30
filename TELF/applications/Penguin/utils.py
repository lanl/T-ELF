import warnings

def create_text_query(attributes, targets, join):
    """
    Constructs a MongoDB query to search for specified target terms within given attributes of documents.

    This method generates a query that searches for each term in 'targets' within each field listed in 
    'attributes'. The search is case-insensitive and looks for partial matches of the target terms within 
    the attribute values. The relationship between different target terms can be specified using `join`. 
    This can be 'AND' or 'OR'. Relations between different attributes are always joined with 'OR'. For
    example, if `attributes` is ['title', 'abstract'] then the query will look for target text in either
    the title or the abstract.

    Parameters:
    -----------
    attributes: [str]
        A list of attribute names (fields) within the documents to search for the target terms.
    targets: [str]
        A list of target terms to search for within the specified attributes.
    join: str
        The logical join operator to combine the search queries for different attributes. 
        Supported values are 'OR' and 'AND'.

    Returns:
    --------
    dict
        A MongoDB query dictionary ready to be used with find() or similar methods.

    Notes:
    ------
    - The search is case-insensitive due to the '$options': 'i' flag in the regex queries.
    - Partial matches are allowed, meaning that documents will be matched if they contain 
      any part of the target terms within the specified attributes.
    """
    queries = []
    for attr in attributes:
        attr_queries = []
        for term in targets:
            attr_queries.append({attr: {'$regex': f'.*{term}.*', '$options': 'i'}})
        attr_queries = {f'${join.lower()}': attr_queries}
        queries.append(attr_queries)

    if len(queries) == 1:
        return queries[0]
    else: 
        return {'$or': queries}
    
def create_general_query(attributes, targets, join):
    queries = []
    for attr in attributes:
        attr_queries = []
        for term in targets:
            attr_queries.append({attr: {'$regex': f'.*{term}.*', '$options': 'i'}})
        attr_queries = {f'${join.lower()}': attr_queries}
        queries.append(attr_queries)

    if len(queries) == 1:
        return queries[0]
    else: 
        return {'$or': queries} 
