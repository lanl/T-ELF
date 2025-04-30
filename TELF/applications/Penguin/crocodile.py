#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: crocodile.py
Project: Penguin
Author: Nick Solovyev
Email: nks@lanl.gov
Date: 01/22/2024

Description:
    This file contains useful procesing functions for converting XML and JSON paper data
    from Scopus and S2 into more compact JSON for storing in a MongoDB instance.
"""
import os
import json
import pathlib
import warnings
import xmltodict
import pandas as pd
pd.options.mode.chained_assignment = None

from collections import Counter
from ...helpers.data_structures import get_from_dict, try_int
from ...helpers.frames import (merge_frames_simple, match_frames_text, reorder_and_add_columns, drop_duplicates)
from ...helpers.data_structures import match_lists

# map for GeoNames place codes to geographical location (country)
# scopus stores country information as GeoNames links so this map is needed to convert links to country strings
IDK = 'Unknown'
COUNTRY_CODES = {
    '1814991': 'China',
    '6252001': 'United States',
    '1861060': 'Japan',
    '1835841': 'South Korea',
    '2635167': 'United Kingdom',
    '6695072': 'European Union',
    '1819730': 'Hong Kong',
    '2077456': 'Australia',
    '102358': 'Saudi Arabia',
    '1880251': 'Singapore',
    '2921044': 'Germany',
    '1668284': 'Taiwan',
    '2510769': 'Spain',
    '6251999': 'Canada',
    '1821275': 'Macao',
    '2017370': 'Russia',
    '798544': 'Poland',
    '2661886': 'Sweden',
    '660013': 'Finland',
    '1605651': 'Thailand',
    '3469034': 'Brazil',
    '1269750': 'India',
    '3017382': 'France',
    '2750405': 'Netherlands',
    '1733045': 'Malaysia',
    '357994': 'Egypt',
    '2782113': 'Austria',
    '719819': 'Hungary',
    '2802361': 'Belgium',
    '2658434': 'Switzerland',
    '3077311': 'Czechia',
    '289688': 'Qatar',
    '1227603': 'Sri Lanka',
    '3144096': 'Norway',
    '3175395': 'Italy',
    '294640': 'Israel',
    '3895114': 'Chile',
    '2264397': 'Portugal',
    '2623032': 'Denmark',
    '2963597': 'Ireland',
    '1562822': 'Vietnam',
    '1168579': 'Pakistan',
    '953987': 'South Africa',
    '3996063': 'Mexico',
    '290557': 'United Arab Emirates',
    '3190538': 'Slovenia',
    '130758': 'Iran',
    '298795': 'Turkey',
    '732800': 'Bulgaria',
    '2186224': 'New Zealand',
    '453733': 'Estonia',
    '3057568': 'Slovakia',
    '3202326': 'Croatia',
    '6290252': 'Serbia',
    '2960313': 'Luxembourg',
    '690791': 'Ukraine',
    '3865483': 'Argentina',
    '390903': 'Greece',
    '1643084': 'Indonesia',
    '798549': 'Romania',
    '337996': 'Ethiopia',
    '1522867': 'Kazakhstan',
    '3932488': 'Peru',
    '3474415': 'South Georgia and South Sandwich Islands',
    '192950': 'Kenya',
    '1694008': 'Philippines',
    '927384': 'Malawi',
    '174982': 'Armenia',
    '99237': 'Iraq',
    '2629691': 'Iceland',
    '3686110': 'Colombia',
    '630336': 'Belarus',
    '2215636': 'Libya',
    '614540': 'Georgia',
    '1820814': 'Brunei',
    '286963': 'Oman',
    '1210997': 'Bangladesh',
    '3439705': 'Uruguay',
    '597427': 'Lithuania',
    '3624060': 'Costa Rica',
    '783754': 'Albania',
    '2542007': 'Morocco',
    '248816': 'Jordan',
    '2589581': 'Algeria',
    '146669': 'Cyprus',
    '226074': 'Uganda',
    '285570': 'Kuwait',
    '3658394': 'Ecuador',
    '3355338': 'Namibia',
    '458258': 'Latvia',
    '2328926': 'Nigeria',
    '3042058': 'Liechtenstein',
    '2993457': 'Monaco',
    '895949': 'Zambia',
    '1512440': 'Uzbekistan',
    '272103': 'Lebanon',
    '3489940': 'Jamaica',
    '3572887': 'Bahamas',
    '878675': 'Zimbabwe',
    '3437598': 'Paraguay', 
    '2464461': 'Tunisia',
    '934292': 'Mauritius',
    '1282988': 'Nepal',
}


def process_scopus_json(file_path, output_dir):
    """ 
    Standardize the input from a Scopus JSON file
    
    This function take a single JSON file output by iPenguin.Scopus. The function then 
    converts the JSON to a unified format to be stored in the Penguin MongoDB database.
    The processed file contents will be saved as EID.json in `output_dir` where EID is
    the Scopus paper ID.
    
    Parameters:
    -----------
    file_path: str, pathlib.Path
        The path to the input Scopus JSON file
    output_dir: str, pathlib.Path, None
        The directory at which to save the processed JSON file. If None, the contents 
        will not be saved and will be returned by the function
    
    Returns:
    --------
    None, dict
        If `output_dir` is provided then the standardized file contents will be saved at 
        the defined directory. Otherwise the results will be returned from this function
        as a a dict. 
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    
    fn = file_path.stem
    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            doc = json.load(fh)
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f'JSONDecodeError for {file_path.name}')
        
    # get the core information from the document
    eid = get_from_dict(doc, ['coredata', 'eid'])
    doi = get_from_dict(doc, ['coredata', 'prism:doi'])
    try:
        doi = doi.lower()
    except:
        pass
    
    title = get_from_dict(doc, ['coredata', 'dc:title'])
    abstract = get_from_dict(doc, ['coredata', 'dc:description'])
    year = try_int(get_from_dict(doc, ['item', 'bibrecord', 'head', 'source', 'publicationyear', '@first']))
    publication_name = get_from_dict(doc, ['coredata', 'prism:publicationName'])
    subject_areas = get_from_dict(doc, ['subject-areas', 'subject-area'])
    funding = get_from_dict(doc, ['item', 'xocs:meta', 'xocs:funding-list'])
    
    # author, affiliation, keywords
    bibrecord = get_from_dict(doc, ['item', 'bibrecord'])
    
    # save new output JSON file
    output = {
        'eid': eid,
        'doi': doi,
        'title': title,
        'abstract': abstract,
        'year': year,
        'publication_name': publication_name,
        'subject_areas': subject_areas,
        'funding': funding,
        'bibrecord': bibrecord,
        'tags': [],
    }
    
    if output_dir is None:
        return output
    with open(os.path.join(output_dir, f'{fn}.json'), 'w') as fh: 
        json.dump(output, fh)
        
        
def process_scopus_xml(file_path, output_dir):
    """ 
    Standardize the input from a Scopus XML file
    
    This function take a single XML file purchased from Scopus. The function then 
    converts the XML to a unified format to be stored in the Penguin MongoDB database.
    The processed file contents will be saved as EID.json in `output_dir` where EID is
    the Scopus paper ID.
    
    Parameters:
    -----------
    file_path: str, pathlib.Path
        The path to the input Scopus XML file
    output_dir: str, pathlib.Path, None
        The directory at which to save the processed JSON file. If None, the contents 
        will not be saved and will be returned by the function
    
    Returns:
    --------
    None, dict
        If `output_dir` is provided then the standardized file contents will be saved at 
        the defined directory. Otherwise the results will be returned from this function
        as a a dict. 
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    
    fn = file_path.stem
    with open(file_path, 'r', encoding='utf-8') as fh:
        my_xml = fh.read()
        doc = xmltodict.parse(my_xml)['xocs:doc']

    # get the basic information about the document
    eid = get_from_dict(doc, ['xocs:meta', 'xocs:eid'])
    doi = get_from_dict(doc, ['xocs:meta', 'xocs:doi'])
    try:
        doi = doi.lower()
    except:
        pass

    title = get_from_dict(doc, ['xocs:item', 'item', 'bibrecord', 'head', 'citation-title', 'titletext'])
    abstract = get_from_dict(doc, ['xocs:item', 'item', 'bibrecord', 'head', 'abstracts', 'abstract', 'ce:para'])
    year = try_int(get_from_dict(doc, ['xocs:item', 'item', 'bibrecord', 'head', 'source', 'publicationyear', '@first']))
    
    # sometimes abstract or title are one structure deeper
    if isinstance(title, dict):  
        title = title.get('#text')
    if isinstance(abstract, dict):  
        abstract = abstract.get('#text')
    
    # publication name
    publication_name = get_from_dict(doc, ['xocs:meta', 'xocs:srctitle'])
    
    # subject areas
    subject_area_abbrevs = get_from_dict(doc, ['xocs:meta', 'xocs:subjareas', 'xocs:subjarea'])
    if subject_area_abbrevs is None:
        subject_area_abbrevs = []
    subject_areas = [{
        '@_fa': 'true', 
        '$': None, 
        '@code': None, 
        '@abbrev': abbrev 
    } for abbrev in subject_area_abbrevs]
        
    # funding 
    funding = get_from_dict(doc, ['xocs:meta', 'xocs:funding-list'])
    
    # author, affiliation, keywords
    bibrecord = get_from_dict(doc, ['xocs:item', 'item', 'bibrecord'])
    
    # save new output JSON file
    output = {
        'eid': eid,
        'doi': doi,
        'title': title,
        'abstract': abstract,
        'year': year,
        'publication_name': publication_name,
        'subject_areas': subject_areas,
        'funding': funding,
        'bibrecord': bibrecord,
        'tags': [],
    }

    if output_dir is None:
        return output
    with open(os.path.join(output_dir, f'{fn}.json'), 'w') as fh: 
        json.dump(output, fh)
        
        
def process_s2_json(file_path, output_dir):
    """ 
    Standardize the input from an S2 JSON file
    
    This function take a single JSON file output by iPenguin.SemanticScholar. The function then 
    converts the JSON to a unified format to be stored in the Penguin MongoDB database.
    The processed file contents will be saved as S2ID.json in `output_dir` where S2ID is
    the S2 paper ID.
    
    Parameters:
    -----------
    file_path: str, pathlib.Path
        The path to the input S2 JSON file
    output_dir: str, pathlib.Path, None
        The directory at which to save the processed JSON file. If None, the contents 
        will not be saved and will be returned by the function
    
    Returns:
    --------
    None, dict
        If `output_dir` is provided then the standardized file contents will be saved at 
        the defined directory. Otherwise the results will be returned from this function
        as a a dict. 
    """
    with open(file_path, 'r') as fh:
        try:
            output = json.load(fh)
        except json.JSONDecodeError as e:
            print(f'Failed to decode JSON from {fn!r}: {e}', file=sys.stderr)
        except Exception as e:
            print(f'An error occurred while inserting data from {fn!r}: {e}', file=sys.stderr)
    
    # remove pre-computed embedding so its not taking up space in mongo
    if 'embedding' in output:
        _ = output.pop('embedding')

    # make sure DOIs are lowercase
    if 'externalIds' in output:
        if 'DOI' in output['externalIds']:
            doi = output['externalIds']['DOI']
            if isinstance(doi, str):
                output['externalIds']['DOI'] = doi.lower()
        
    # add empty list for tags
    output['tags'] = []
        
    if output_dir is None:
        return output
    with open(os.path.join(output_dir, f'{fn}.json'), 'w') as fh: 
        json.dump(output, fh)
        

def parse_scopus_funding(funding_info):
    """ 
    Parse the Scopus funding information if possible

    Parameters:
    -----------
    funding_info: dict, list
        Raw Scopus funding information
    
    Returns:
    --------
    None, list
        Processed funding info for SLIC DataFrame
    """
    if not funding_info:
        funding = None
    else:
        funding = []
        if not isinstance(funding_info, list):
            funding_info = [funding_info]

        for f in funding_info:
            funding_id = get_from_dict(f, ['xocs:funding-id'])
            funding_agency = get_from_dict(f, ['xocs:funding-agency'])
            funding_country = get_from_dict(f, ['xocs:funding-agency-country'])

            if funding_id and funding_agency:
                funding.append({
                    'funding_id': funding_id,
                    'funding_agency': funding_agency,
                    'funding_country': funding_country
                })
        if not funding:
            funding = None
    return funding


def parse_scopus_affiliations(auth_info):
    """
    Parse the Scopus author and affiliation information if possible

    Parameters:
    -----------
    auth_info: dict, list
        Raw Scopus author/affiliation information
    
    Returns:
    --------
    None, list
        Processed affiliation and author info for SLIC DataFrame
    """
    author_names = {}
    affiliations = {}
    for entry in auth_info:    
        country = get_from_dict(entry, ['affiliation', 'country'])
        organization = get_from_dict(entry, ['affiliation', 'organization'])
        affiliation_id = get_from_dict(entry, ['affiliation', 'affiliation-id'])

        if isinstance(organization, list) and isinstance(affiliation_id, dict):
            # if the last entry into the affiliation structure does not exist, this is a non-standard data structure
            # this dictionary could contain the organization or it could contain something else
            # there is no easy way to validate data in this structure so best to skip it
            if not organization[-1]:
                organization = None
            else:
                if all(isinstance(x, str) for x in organization):
                    if 'ltd' in organization[-1].lower():  
                        organization = organization[-2]
                    else:
                        # this line is causing problems in Scopus papers where the affiliation is stored as a list
                        # where the first entry is the main entry and the last entry is something like the dept. Going
                        # to try replacing this with a joined string of the entire affiliation structure
                        organization = ', '.join(organization)
                else:
                    if 'ltd' in organization[-1].get('$', '').lower():  
                        organization = organization[-2]  # take the company name (second to last element)
                    else:
                        # this line is causing problems in Scopus papers where the affiliation is stored as a list
                        # where the first entry is the main entry and the last entry is something like the dept. Going
                        # to try replacing this with a joined string of the entire affiliation structure
                        ##organization = organization[-1]
                        try:
                            organization = {'$': ', '.join([next(iter(x.values())) for x in organization])}
                        except:
                            organization = [None]

        if not isinstance(organization, list):
            organization = [organization]
            affiliation_id = [affiliation_id]

        if affiliation_id is None or organization[0] is None:
            continue
        aff_matches = match_lists(affiliation_id, organization)
        for aff_id, org in aff_matches.items():
            
            # setup affiliation structure
            aff_info = {
                'country': country or 'Unknown',
                'name': org,
                'authors': set(),
            }

            authors = entry.get('author')
            if not authors:  # no author information, affiliation useless, skip
                continue
            if not isinstance(authors, list):
                authors = [authors]

            for auth in authors:
                auth_id = auth.get('@auid')
                if auth_id:
                    auth_id = str(auth_id)
                    auth_name = auth.get('ce:indexed-name')
                    author_names[auth_id] = auth_name
                    aff_info['authors'].add(auth_id)

            if aff_id is not None:
                if aff_id not in affiliations:
                    affiliations[aff_id] = aff_info
                    affiliations[aff_id]['name'] = [affiliations[aff_id]['name']]
                else:
                    affiliations[aff_id]['name'].append(aff_info['name'])
                    affiliations[aff_id]['authors'] |= set(aff_info['authors'])
                    
                    
    ids = list(author_names.keys()) or None
    if ids:
        ids = ';'.join(ids)
    names = list(author_names.values()) or None
    if names:
        names = ';'.join(names)

    if not affiliations:  # set affiliations to None if empty dictionary 
        affiliations = None
    else:  # otherwise sort affiliations by number of authors per affiliation
        for aff_id in affiliations:
            organization, _ = Counter(affiliations[aff_id]['name']).most_common(1)[0]
            if organization is None:
                organization = 'Unknown'
            authors = list(affiliations[aff_id]['authors'])
            affiliations[aff_id]['name'] = organization
            affiliations[aff_id]['authors'] = authors 
        affiliations = dict(sorted(affiliations.items(), key=lambda x: len(x[1]['authors']), reverse=True))
    return names, ids, affiliations


def form_scopus_df(data):
    """
    Given an iterable of Penguin Scopus data, form a pandas DataFrame
    
    Parameters
    ----------
    data: Iterable
        The files from which the DataFrame should be generated

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame
    """
    dataset= {
        "eid":[],
        "doi":[],
        "title":[],
        "year":[],
        "abstract":[],
        "authors":[],
        "author_ids":[],
        "affiliations":[],
        "funding":[],
        "PACs":[],
        "publication_name": [],
        "subject_areas": [],
        "num_citations": [],
    }

    if data:
        for doc in data:
                
            # get the basic information about the document
            eid = doc.get('eid')
            doi =  doc.get('doi')
            title =  doc.get('title')
            year = try_int(doc.get('year'))
            abstract = doc.get('abstract')
    
            # get the publication name (journal, conference, etc)
            publication_name = doc.get('publication_name')
        
            # get the number of citations
            num_citations = try_int(doc.get('num_citations'))
            
            # get the subject areas
            subject_areas = doc.get('subject_areas')
            if subject_areas:
                subject_areas = ';'.join({x.get('@abbrev', 'Unknown') for x in subject_areas})
            else:
                subject_areas = None
                
            # get author, author id, and affiliation information
            authors = None
            author_ids = None
            affiliations = None
            auth_info = get_from_dict(doc, ['bibrecord', 'head', 'author-group'])
            if auth_info:
                if not isinstance(auth_info, list):
                    auth_info = [auth_info]
                authors, author_ids, affiliations = parse_scopus_affiliations(auth_info)
    
            # get funding information (if possible)
            funding = None
            funding_info = get_from_dict(doc, ['funding', 'xocs:funding'])
            if funding_info:
                funding = parse_scopus_funding(funding_info)
    
            # get keywords (if possible)
            keywords = get_from_dict(doc, ['bibrecord', 'head', 'citation-info', 'author-keywords', 'author-keyword'])
            if keywords:
                if isinstance(keywords, dict):
                    keywords = [x.strip() for x in keywords.get('$','').split(';')]
                elif isinstance(keywords[0], dict):
                    keywords = [x.get('$','') for x in keywords]
                keywords = [x for x in keywords if x]
                keywords = ';'.join(keywords)
    
    
            # add everything to dataset
            dataset['eid'].append(eid)
            dataset['doi'].append(doi)
            dataset['title'].append(title)
            dataset['year'].append(year)
            dataset['abstract'].append(abstract)
            dataset['authors'].append(authors)
            dataset['author_ids'].append(author_ids)
            dataset['affiliations'].append(affiliations)
            dataset['funding'].append(funding)
            dataset['PACs'].append(keywords)
            dataset['publication_name'].append(publication_name)
            dataset['subject_areas'].append(subject_areas)
            dataset['num_citations'].append(num_citations)
    
    # form the dataframe
    df = pd.DataFrame.from_dict(dataset)
    
    # adjust funding information 
    adjusted_funding = []
    for index, row in df.iterrows():
        funding = row['funding']
        if not funding or isinstance(funding, float):
            adjusted_funding.append(None)
            continue

        adjusted_fobjs = {}
        for i, fobj in enumerate(funding):
            fids = fobj['funding_id']
            if not isinstance(fids, list):
                fids = [fids]
            fname = fobj['funding_agency']
            fcountry = fobj.get('funding_country', 'Unknown')
            if not fcountry:
                fcountry = 'Unknown'

            if 'http' in fcountry:
                if fcountry[-1] == '/':
                    fcountry = fcountry[0:-1]
                fcountry = fcountry.rsplit('/')[-1]
                fcountry = COUNTRY_CODES.get(fcountry, 'Unknown')

            if fname not in adjusted_fobjs:
                adjusted_fobjs[fname] = {
                    'funding_country': fcountry,
                    'funding_ids': set(),
                }

            if isinstance(fids[0], dict):
                fids = [x['$'] for x in fids if x is not None]
                
            adjusted_fobjs[fname]['funding_ids'] |= set(fids)
        for k in adjusted_fobjs:
            adjusted_fobjs[k]['funding_ids'] = list(adjusted_fobjs[k]['funding_ids'])
        adjusted_funding.append(adjusted_fobjs)

    df['funding'] = adjusted_funding
    return df


def form_s2_df(data):
    """
    Given an iterable of Penguin S2 data, form a pandas DataFrame
    
    Parameters
    ----------
    data: Iterable
        The files from which the DataFrame should be generated

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame
    """
    dataset = {
        's2id': [],
        'doi': [],
        'year': [],
        'title': [],
        'abstract': [],
        's2_authors': [],
        's2_author_ids': [],
        'citations': [],
        'references': [],
        'num_citations': [],
        'num_references': [],
    }

    for doc in data:

        # get the basic information about the document
        s2id = doc.get('paperId')
        doi = get_from_dict(doc, ['externalIds', 'DOI'])
        title = doc.get('title')
        year = try_int(doc.get('year'))
        abstract = doc.get('abstract')

        # get author information
        authors = []
        author_ids = []
        for auth in doc.get('authors'):
            auth_id = auth.get('authorId')
            auth_name = auth.get('name', 'Unknown')
            if auth_id:
                authors.append(auth_name)
                author_ids.append(auth_id)
        
        if authors:
            authors = ';'.join(authors)
            author_ids = ';'.join(author_ids)
        else:
            authors = None
            author_ids = None
        
        # get citations
        citations = None
        num_citations = None
        if doc.get('citations'):
            citations = [x.get('paperId', None) for x in doc['citations']]
            citations = [str(x) for x in citations if x is not None]
            if not citations:
                citations = None
            else: 
                num_citations = len(citations)
                citations = ';'.join(citations)

        # get references
        references = None
        num_references = None
        if doc.get('references'):
            references = [x.get('paperId') for x in doc['references']]
            references = [str(x) for x in references if x is not None]
            if not references:
                references = None
            else:
                num_references = len(references)
                references = ';'.join(references)
       
        # add everything to dataset
        dataset['s2id'].append(s2id)
        dataset['doi'].append(doi)
        dataset['title'].append(title)
        dataset['year'].append(year)
        dataset['abstract'].append(abstract)
        dataset['s2_authors'].append(authors)
        dataset['s2_author_ids'].append(author_ids)  
        dataset['citations'].append(citations)
        dataset['num_citations'].append(num_citations)
        dataset['references'].append(references)
        dataset['num_references'].append(num_references)
        
    # turn into DataFrame and return
    return pd.DataFrame.from_dict(dataset)


def form_df(scopus_df, s2_df, df_order=None):
    """
    Form a SLIC DataFrame from Scopus and S2 frames
  
    Parameters
    ----------
    scopus_df: pd.DataFrame
        The Scopus DataFrame
    s2_df: pd.DataFrame
        The S2 DataFrame
    df_order: list, (optional)
        The order of columns for the output DataFrame. If None, default columns are used

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame
    """
    if df_order is None:
        df_order = [
            'eid', 
            's2id', 
            'doi', 
            'title', 
            'abstract',
            'year',
            'authors',
            'author_ids', 
            'affiliations', 
            'funding',
            'PACs', 
            'publication_name',
            'subject_areas',
            's2_authors',
            's2_author_ids',
            'citations',
            'num_citations',
            'references',
            'num_references',
        ]

    if scopus_df.empty:
        return reorder_and_add_columns(s2_df, df_order)
    if s2_df.empty:
        return reorder_and_add_columns(scopus_df, df_order)
    
    # get papers with doi defined
    s2_df_doi_missing = s2_df[s2_df['doi'].isna()]
    s2_df_doi_not_missing = s2_df[~s2_df['doi'].isna()]
    scopus_df_doi_missing = scopus_df[scopus_df['doi'].isna()]
    scopus_df_doi_not_missing = scopus_df[~scopus_df['doi'].isna()]
    
    # merge based on doi
    s2_df_doi_missing = reorder_and_add_columns(s2_df_doi_missing, df_order)
    scopus_df_doi_missing = reorder_and_add_columns(scopus_df_doi_missing, df_order)
    s2_df_doi_not_missing = reorder_and_add_columns(s2_df_doi_not_missing, df_order)
    scopus_df_doi_not_missing = reorder_and_add_columns(scopus_df_doi_not_missing, df_order)
    doi_merged_df = merge_frames_simple(scopus_df_doi_not_missing, s2_df_doi_not_missing, 'doi')
    
    # get unmatched documents
    _s2_df = pd.concat([
        doi_merged_df.loc[doi_merged_df.eid.isnull()],
        s2_df_doi_missing,
    ])
    
    _scopus_df = pd.concat([
        doi_merged_df.loc[doi_merged_df.s2id.isnull()],
        scopus_df_doi_missing,
    ])
    
    # match based on high precision text matching of titles
    s2_df_text_missing = _s2_df[_s2_df['title'].isna()]
    s2_df_text_not_missing = _s2_df[~_s2_df['title'].isna()]
    scopus_df_text_missing = _scopus_df[_scopus_df['title'].isna()]
    scopus_df_text_not_missing = _scopus_df[~_scopus_df['title'].isna()]
    scopus_df_text_not_missing, s2_df_text_not_missing, match_col = match_frames_text(df1 = scopus_df_text_not_missing,
                                                                                      df2 = s2_df_text_not_missing,
                                                                                      col = 'title',
                                                                                      min_len = 12,
                                                                                      threshold = 0.025)

    # get papers with key text match
    s2_df_key_missing = s2_df_text_not_missing[s2_df_text_not_missing[match_col].isna()]
    s2_df_key_missing.drop(columns=[match_col], inplace=True)
    s2_df_key_not_missing = s2_df_text_not_missing[~s2_df_text_not_missing[match_col].isna()]
    scopus_df_key_missing = scopus_df_text_not_missing[scopus_df_text_not_missing[match_col].isna()]
    scopus_df_key_missing.drop(columns=[match_col], inplace=True)
    scopus_df_key_not_missing = scopus_df_text_not_missing[~scopus_df_text_not_missing[match_col].isna()]

    # merge based on text key
    if not scopus_df_key_not_missing.empty and not s2_df_key_not_missing.empty:
        text_merged_df = merge_frames_simple(scopus_df_key_not_missing, s2_df_key_not_missing, match_col)
    elif not scopus_df_key_not_missing.empty:
        text_merged_df = scopus_df_key_not_missing.copy()
    elif not s2_df_key_not_missing.empty:
        text_merged_df = s2_df_key_not_missing.copy()
    else:
        text_merged_df = s2_df_key_not_missing.copy()
    
    text_merged_df.drop(columns=[match_col], inplace=True)
    
    #combine
    with warnings.catch_warnings():
        # TODO: pandas >= 2.1.0 has a FutureWarning for concatenating DataFrames with Null entries
        warnings.filterwarnings("ignore", category=FutureWarning)
        merged_df = pd.concat([
            doi_merged_df,
            text_merged_df,
            s2_df_text_missing,
            scopus_df_text_missing,
            s2_df_key_missing,
            scopus_df_key_missing,
        ], ignore_index=True)
    
    # drop duplicates
    merged_df = drop_duplicates(merged_df, 'eid')
    merged_df = drop_duplicates(merged_df, 's2id')
    return merged_df
