import re
import json
import pathlib
import warnings
import pandas as pd

def extract_doi(s):
    return re.sub(r'^https://doi\.org/', '', s)


def extract_year(iso_date):
    match = re.match(r'^(\d{4})-', iso_date)
    return int(match.group(1)) if match else None


def process_single_author(s):
    
    # 1. extract the name
    name_match = re.match(r'([\w\s,\.]+)', s)
    name = name_match.group(1).strip() if name_match else 'NAN'
    
    # 2 extract the ID (if possible)
    orcid_match = re.search(r'ORCID:(\d+)', s)
    orcid = orcid_match.group(1) if orcid_match else 'NAN'
    
    # 3. extract the affiliation information (if possible)
    aff_info = {}
    aff_section = re.search(r'\[(.*?)\]', s)
    if aff_section:
        affiliations = aff_section.group(1)
        
        # try to split on semicolon to account for multiple affiliations
        affiliations = [x.strip() for x in affiliations.split(';')]
        for affiliation in affiliations:
            
            aff_match = re.match(r'^([\w\s\.]+?)(?: \([\w\s]+\))?,.*\(([\w\s]+)\)', affiliation)
            if aff_match:
                aff, country = aff_match.groups()
                if aff not in aff_info:
                    aff_info[aff] = {'country': country, 'names': []}
                aff_info[aff]['names'].append(name)
    
    return name, orcid, aff_info


def process_authors(auth_list):
    authors = []
    author_ids = []
    affiliations = {}

    for auth_str in auth_list:
        name, orcid, aff_info = process_single_author(auth_str)
        authors.append(name)
        author_ids.append(orcid)

        for org, data in aff_info.items():
            if org not in affiliations:
                affiliations[org] = {'country': data['country'], 'names': []}
            affiliations[org]['names'].extend(data['names'])

    authors = ";".join(authors)
    author_ids = ";".join(author_ids)
    return authors, author_ids, affiliations


def form_df(files):
    """
    Given a list of file names corresponding to OSTI JSON data, form a pandas DataFrame from 
    said file names
    
    Parameters
    ----------
    path : str, pathlib.Path
        The path to the directory where the JSON files are stored
    files : list
        The file names from which the DataFrame should be generated

    Returns
    -------
    pd.DataFrame
        Resulting DataFrame
    """
    dataset= {
        "osti_id":[],
        "doi":[],
        "title":[],
        "year":[],
        "abstract":[],
        "authors":[],
        "author_ids":[],
        "affiliations":[],
        "country_publication":[],
        "report_number":[],
        "doe_contract_number":[],
        "publisher":[],
        "language":[],
    }
    
    for f in files:
        try:
            with open(f, 'r') as fh:
                doc = json.load(fh)
        except json.JSONDecodeError:
            f = pathlib.Path(f)
            warnings.warn(f'[OSTI]: JSONDecodeError for {f.name}, skipping', RuntimeWarning)
            continue
            
        # get the basic information about the document
        osti_id = doc.get('osti_id')
        doi =  doc.get('doi')
        if doi is not None:
            doi = extract_doi(doi)
        
        title = doc.get('title')
        publication_date = doc.get('publication_date', '')
        year = extract_year(publication_date)
        abstract = doc.get('description')

        # get author, author id, and affiliation information
        authors = None
        author_ids = None
        affiliations = None
        auth_list = doc.get('authors', [])
        authors, author_ids, affiliations = process_authors(auth_list)
        if not affiliations:
            affiliations = None
        
        # get some OSTI specific information
        country_publication = doc.get('country_publication')
        report_number = doc.get('report_number')
        doe_contract_number = doc.get('doe_contract_number')
        publisher = doc.get('publisher')
        language = doc.get('language')
        
        # add everything to dataset
        dataset['osti_id'].append(osti_id)
        dataset['doi'].append(doi)
        dataset['title'].append(title)
        dataset['year'].append(year)
        dataset['abstract'].append(abstract)
        dataset['authors'].append(authors)
        dataset['author_ids'].append(author_ids)
        dataset['affiliations'].append(affiliations)
        dataset['country_publication'].append(country_publication)
        dataset['report_number'].append(report_number)
        dataset['doe_contract_number'].append(doe_contract_number)
        dataset['publisher'].append(publisher)
        dataset['language'].append(language)
    
    # form the dataframe
    df = pd.DataFrame.from_dict(dataset)
    return df