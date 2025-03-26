import json
import pathlib
import warnings
import pandas as pd

from ..utils import try_int, get_from_dict

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

def gen_chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]
        
        
def parse_funding(funding_info):

    # add the funding info if possible
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


def match_lists(list_a, list_b, key_a='@afid', key_b='$'):
    result = {}
    len_a = len(list_a)
    len_b = len(list_b)
    
    for i in range(max(len_a, len_b)):
        value_a = list_a[i][key_a] if i < len_a and isinstance(list_a[i], dict) and key_a in list_a[i] else None
        value_b = list_b[i][key_b] if i < len_b and isinstance(list_b[i], dict) and key_b in list_b[i] else None
        result[value_a] = value_b        
    return result


def parse_affiliation(affiliations, authors):
    affiliation_map = {}
    authors_map = ''
    author_ids= ''
    if affiliations:
        # print(affiliations)
        if isinstance(affiliations,dict):
            affiliations = [affiliations]

        for affiliaton in affiliations:
            if isinstance(affiliaton,dict):
                country = affiliaton.get('affiliation-country')
                affiliation_id = affiliaton.get('@id')
                name = affiliaton.get('affilname')
                affiliation_map[affiliation_id] = {'country':country,
                                                   'name':name,
                                                   'authors':[]}
            else:
                print(affiliaton)

    if isinstance(authors,dict):
        authors = [authors]

    if isinstance(authors,list):
        for author in authors:
            name = get_from_dict(author, ['preferred-name', 'ce:indexed-name'])
            auth_id = author.get('@auid', IDK)
            authors_map += name + ';'
            author_ids += auth_id + ';'

            auth_affs = author.get('affiliation')
            if auth_affs:
                if isinstance(auth_affs, dict):
                    auth_affs = [auth_affs]

                for aff in auth_affs:
                    auth_aff_id = aff.get('@id')
                    if affiliation_map.get(auth_aff_id):
                        affiliation_map[auth_aff_id]['authors'].append(auth_id)
                    else:
                        affiliation_map[auth_aff_id] = {'authors': [auth_id]}
                        print("WARNING: Affiliation for author doesnt exist--", author, aff)

            else:
                if affiliation_map.get(IDK):
                    affiliation_map[IDK].append((auth_id, name))
                else:
                    affiliation_map[IDK]=[(auth_id, name)]

    authors_map = authors_map[:-1]
    author_ids = author_ids[:-1]
    return affiliation_map, authors_map, author_ids

def form_df(files):
    """
    Given a list of file names corresponding to Scopus XML data, form a pandas DataFrame from 
    said file names
    
    Parameters
    ----------
    path : str, pathlib.Path
        The path to the directory where the XML files are stored
    files : list
        The file names from which the DataFrame should be generated

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
    
    for f in files:
        try:
            with open(f, 'r') as fh:
                doc = json.load(fh)
        except json.JSONDecodeError:
            f = pathlib.Path(f)
            warnings.warn(f'[Scopus]: JSONDecodeError for {f.name}, skipping', RuntimeWarning)
            continue
            
        # get the basic information about the document
        eid = get_from_dict(doc, ['coredata', 'eid'])
        doi = get_from_dict(doc, ['coredata', 'prism:doi'])
        title = get_from_dict(doc, ['coredata', 'dc:title'])
        year = try_int(get_from_dict(doc, ['item', 'bibrecord', 'head', 'source', 'publicationyear', '@first']))
        abstract = get_from_dict(doc, ['coredata', 'dc:description'])
        if isinstance(abstract, dict):  # sometimes abstract is one structure deeper
            abstract = abstract.get('#text')

        # get the publication name (journal, conference, etc)
        publication_name = get_from_dict(doc, ['coredata', 'prism:publicationName'])
            
        # get the subject areas
        subject_areas = get_from_dict(doc, ['subject-areas', 'subject-area'])
        if subject_areas:
            subject_areas = ';'.join([x.get('$', 'Unknown') for x in subject_areas])
        else:
            subject_areas = None
        
        # get the number of citations
        num_citations = get_from_dict(doc, ['coredata', 'citedby-count'])
        if num_citations is None:
            num_citations = 0
        else:
            num_citations = int(num_citations)
            
        # get author, author id, and affiliation information
        authors = None
        author_ids = None
        affiliations = None

        affiliation_new = get_from_dict(doc, ['affiliation'])
        authors_new = get_from_dict(doc, ['authors', 'author'])
        affiliations, authors, author_ids = parse_affiliation(affiliation_new, authors_new)


        # get funding information (if possible)
        funding = None
        funding_info = get_from_dict(doc, ['item', 'xocs:meta', 'xocs:funding-list', 'xocs:funding'])
        if funding_info:
            funding = parse_funding(funding_info)

        # get keywords (if possible)
        keywords = get_from_dict(doc, ['authkeywords', 'author-keyword'])
        if keywords:
            if isinstance(keywords, dict):
                keywords = [x.strip() for x in keywords['$'].split(';')]
            elif isinstance(keywords[0], dict):
                keywords = [x['$'] for x in keywords]
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
