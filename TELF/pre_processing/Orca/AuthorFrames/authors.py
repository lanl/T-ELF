import sys
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed
from ....helpers.host import verify_n_jobs
from ....helpers.data_structures import get_from_dict

class ScopusAuthorsFrame:
    """ 
    Process Scopus JSON data to create an author's DataFrame. This DataFrame can be passed to DAF
    to detect duplicate Scopus author IDs
    """
    def __init__(self, n_jobs=-1, verbose=10):
        
        """
        Init ScopusAuthorsFrame object

        Parameters
        ----------
        n_jobs: int, optional
            Number of parallel processes. The default is -1.
        verbose: int, optional
            Verbosity level. The default is 10.

        Returns
        -------
        None
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    
    def __call__(self, data):
        return self.create(data)
    
    
    def create(self, data):
        if self.verbose:
            print('[ScopusAuthorsFrame]: Processing input documents. . .', file=sys.stderr)
        if 1 < self.n_jobs:
            jobs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(ScopusAuthorsFrame.process_doc)(d) for d in data)
        else:
            jobs = []
            for d in tqdm(data, total=len(data), disable=not self.verbose):
                jobs.append(d)

        df, collab, papers = self._merge_processed_docs(jobs)
        names, emails, orcids, affiliations = self._process_author_info(df, collab, papers)
        auth_df =  self._finalize_results(names, emails, orcids, affiliations, papers, collab)
        return auth_df
    
    
    def _merge_processed_docs(self, jobs):
        collab = {}
        papers = {}
        dataset= {
            "aid":[],
            "name":[],
            "email":[],
            "orcid":[],
            "affiliation":[],
        }

        if self.verbose:
            print('[ScopusAuthorsFrame]: Merging processed documents. . .', file=sys.stderr)
        for c, p, d in tqdm(jobs, total=len(jobs), disable=not self.verbose):
            for k,v in c.items():
                if k not in collab:
                    collab[k] = v
                else:
                    collab[k] |= v

            for k,v in p.items():
                if k not in papers:
                    papers[k] = v
                else:
                    papers[k] |= v

            dataset['aid'] += d['aid']
            dataset['name'] += d['name']
            dataset['email'] += d['email']
            dataset['orcid'] += d['orcid']
            dataset['affiliation'] += d['affiliation']

        return pd.DataFrame.from_dict(dataset), collab, papers

    
    def _process_author_info(self, df, collab, papers):
        names = {}
        emails = {}
        orcids = {}
        affiliations = {}
        if self.verbose:
            print('[ScopusAuthorsFrame]: Analyzing author information. . .', file=sys.stderr)
        for aid, name, orcid, email, affiliation in tqdm(zip(df['aid'].to_list(), df['name'].to_list(), df['orcid'].to_list(),
                                                             df['email'].to_list(), df['affiliation'].to_list()),
                                                        total=len(df), disable= not self.verbose):
            if aid not in names:
                names[aid] = name
            else:
                if len(name) > len(names[aid]):
                    names[aid] = name

            if email:
                if aid not in emails:
                    emails[aid] = {email}
                else:
                    emails[aid].add(email)

            if orcid:
                if aid not in orcids:
                    orcids[aid] = {orcid}
                else:
                    orcids[aid].add(orcid)

            if affiliation:
                for year, aff in affiliation:
                    if aid not in affiliations:
                        if year:
                            affiliations[aid] = [(aff, (year, year))]
                        else:
                            affiliations[aid] = [(aff, (0, 10000))]  # invalid min and max years set to be overwritten
                    else:
                        is_found = False
                        for i, (cur_aff, year_range) in enumerate(affiliations[aid]):
                            if cur_aff['afid'] == aff['afid']:
                                min_year, max_year = year_range
                                if year:
                                    min_year = min(min_year, year)
                                    max_year = max(max_year, year)
                                affiliations[aid][i] = (aff, (min_year, max_year))
                                is_found = True

                        if not is_found:
                            affiliations[aid].append((aff, (year, year)))
            
        return names, emails, orcids, affiliations
        
        
    def _finalize_results(self, names, emails, orcids, affiliations, papers, collab):
        dataset= {
            "aid":[],
            "name":[],
            "emails":[],
            "orcid":[],
            "affiliation":[],
            "collaborators":[],
            "papers":[],
        }

        if self.verbose:
            print('[ScopusAuthorsFrame]: Finalizing results. . .', file=sys.stderr)
        for aid in tqdm(names, total=len(names), disable=not self.verbose):
            dataset['aid'].append(aid)
            dataset['name'].append(names[aid])

            if aid in emails:
                dataset['emails'].append(";".join(emails[aid]))
            else:
                dataset['emails'].append(None)

            if aid in orcids:
                dataset['orcid'].append(";".join(orcids[aid]))
            else:
                dataset['orcid'].append(None)

            dataset['papers'].append(papers[aid])  
            if not collab[aid]:
                dataset['collaborators'].append(None) 
            else:
                dataset['collaborators'].append(collab[aid])  

            if aid in affiliations:
                aff_list = []
                for aff, year_range in affiliations[aid]:
                    min_year, max_year = year_range
                    if aff:
                        aff['min_year'] = min_year
                        aff['max_year'] = max_year
                        aff_list.append(aff)

                dataset['affiliation'].append(aff_list)
            else:
                dataset['affiliation'].append(None)

        return pd.DataFrame.from_dict(dataset)

    
    @staticmethod
    def process_doc(doc):
        collab = {}
        papers = {}
        dataset= {
            "aid":[],
            "name":[],
            "email":[],
            "orcid":[],
            "affiliation":[],
        }


        eid = get_from_dict(doc, ['eid'])
        year = get_from_dict(doc, ['year'])
        auth_info = get_from_dict(doc, ['bibrecord', 'head', 'author-group'])

        if not isinstance(auth_info, list):
            auth_info = [auth_info]
        if auth_info:

            # create a set of collaborating ids per paper
            collaborators = set()
            for entry in auth_info:
                auth_entry = get_from_dict(entry, ['author'])
                if auth_entry and not isinstance(auth_entry, list):
                    auth_entry = [auth_entry]
                if auth_entry:
                    for auth in auth_entry:
                        auth_id = get_from_dict(auth, ['@auid'])
                        if not auth_id or auth_id in collaborators:
                            continue                      
                        collaborators.add(auth_id)

                        if auth_id not in papers:
                            papers[auth_id] = {eid}
                        else:
                            papers[auth_id].add(eid)


            # process this set of authors
            seen = {}
            for entry in auth_info:
                auth_entry = get_from_dict(entry, ['author'])
                if auth_entry and not isinstance(auth_entry, list):
                    auth_entry = [auth_entry]
                if auth_entry:
                    for auth in auth_entry:
                        auth_id = get_from_dict(auth, ['@auid'])
                        if not auth_id:
                            continue                      

                        if auth_id not in seen:
                            c_copy = collaborators.copy()
                            c_copy.remove(auth_id)                   
                            if auth_id not in collab:
                                collab[auth_id] = c_copy
                            else:
                                collab[auth_id] |= c_copy

                            seen[auth_id] = {}
                            seen[auth_id]['name'] = get_from_dict(auth, ['ce:indexed-name'])
                            seen[auth_id]['email'] = get_from_dict(auth, ['ce:e-address', '#text'])
                            seen[auth_id]['orcid'] = get_from_dict(auth, ['@orcid'])
                            seen[auth_id]['affiliation'] = []

                        auth_affinfo = {
                            'afid': get_from_dict(entry, ['affiliation', '@afid']),
                            'organization':  get_from_dict(entry, ['affiliation', 'organization']),
                            'country': get_from_dict(entry, ['affiliation', 'country']),
                        }
                        seen[auth_id]['affiliation'].append((year, auth_affinfo))


            for k,v in seen.items():
                dataset['aid'].append(k)
                dataset['name'].append(v['name'])
                dataset['email'].append(v['email'])
                dataset['orcid'].append(v['orcid'])
                dataset['affiliation'].append(v['affiliation'])

        return collab, papers, dataset


    @property
    def n_jobs(self):
        return self._n_jobs
    
    @n_jobs.setter
    def n_jobs(self, n_jobs):
        self._n_jobs = verify_n_jobs(n_jobs)

    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, verbose):
        if isinstance(verbose, bool):
            self._verbose = int(verbose)  # convert False to 0, True to 1
        elif isinstance(verbose, int):
            if verbose < 0:
                raise ValueError('Integer values for `verbose` must be non-negative!')
            self._verbose = verbose
        else:
            raise TypeError('`verbose` should be of type bool or int!')