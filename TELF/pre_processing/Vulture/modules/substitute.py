import re
import warnings
import itertools
import networkx as nx

from TELF.pre_processing.Vulture.modules import VultureModuleBase
from TELF.pre_processing.Vulture.modules import LemmatizeCleaner


class SubstitutionCleaner(VultureModuleBase):
    """
    A text cleaner that finds substrings in the text and replaces them with some other
    substring
    
    The intent of this module is to provide a Subject Matter Expert (SME) the ability to 
    normalize important terms in the vocabulary. Multiple terms can be used to represent
    the same concept. Alternatively, multiple tokens can represent a single term, the meaning
    of which will be destroyed through tokenization. Substitution can normalize these terms
    and also signal to Vulture that these terms do not require cleaning. This means that this 
    module can also be used to "freeze" terms through a reflective substitution, leaving them
    unchanged throughout the cleaning process.
    
    Attributes:
    -----------
    substitution_map: dict 
        A dictionary that stores the substitutions that should occur in the text. The key should
        be a string that needs to be replaced and the value should be the replacement string.
    permute: bool
        A flag that controls whether the keys of the dictionary should be permuted to account for
        whitespace and tokens. For example the string 'foo-bar foo' can be permuted as ['foo bar foo', 
        'foo-bar foo', 'foo bar-foo', 'foo-bar-foo']. This option may be useful when dealing with
        acronyms that use hyphens/whitespace inconsistently.
    lower: bool
        A flag that controls whether the keys in the substitution map should be converted to lowercase.
        Useful when substitution is being used as part of the cleaning pipeline and the term(s) 
        being substituted can take on multiple forms depending on case.
    lemmatize: bool, str
        Controls whether the keys of the substitution dictionary should be lemmatized. If this is a bool
        then the default LemmatizeCleaner backend library is used. If this is a string then it should 
        match one the LemmatizeCleaner backend library options.
    """
    def __init__(self, substitution_map, permute=False, lower=False, lemmatize=False, frozen=None):
        super().__init__(frozen)  # initialize the base class with the preserve
        
        self.module_type = "CLEANER"
        self.lower = lower
        self.permute = permute
        self.lemmatize = lemmatize
        self.substitution_map = substitution_map
        
        
    def __call__(self, document):
        return self.run(document)
        
        
    def run(self, document):
        """
        Run the substitution

        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to perform substitution

        Returns
        -------
        tuple
            Tuple of document id and substituted text
        """
        doc_id, doc_text = document
        flags = re.IGNORECASE if self.lower else 0
        for term, target in self.substitution_map.items():
            if term == target:  # if reflective substition (freezing some important term)
                continue
            if target.strip() == '':  # if using substitution for deletion
                doc_text = re.sub(r'\b{}\b(?=\s|$)'.format(re.escape(term)), target, doc_text, flags=flags)
            else:
                doc_text = re.sub(r'\b{}\b'.format(re.escape(term)), target, doc_text, flags=flags)
        return (doc_id, doc_text)

    
    def _validate_substitutions(self, substitution_map):
        """
        A valid substitution map can be modeled as a bipartite graph with some constraints. 
        This function creates this bipartite graph with the two sets of nodes being target words and
        words to be replaced. If the substitution map is valid, the function returns True. Otherwise,
        the function raises the appropriate error.
        
        Parameters:
        -----------
        substitution_map: dict 
            A dictionary that stores the substitutions that should occur in the text. The key should
            be a string that needs to be replaced and the value should be the replacement string.
        
        Returns:
        --------
        bool
            True if valid, False if dict check fails
            
        Raises:
        -------
        ValueError:
            If substitution_map does not pass bipartite graph check
        """
        dict_is_valid = all(isinstance(key, str) and isinstance(value, str) for key, value in substitution_map.items())
        if not dict_is_valid:
            return False
        
        # set up directed graph for validity check
        G = nx.DiGraph() 

        # add nodes from both sets
        for term, replacement in substitution_map.items():
            # warn user that whitespace in replacement may cause problems/clashes with other vulture modules
            if ' ' in replacement:
                warnings.warn(f'[SubstitutionCleaner]: The replacement "{replacement}" for "{term}" contains whitespace ' \
                               'which may cause problems with tokenization in other vulture modules', RuntimeWarning)
            
            G.add_node(term, bipartite=0)
            G.add_node(replacement, bipartite=1)

        # add directed edges that go from term to be replaced to replacement
        for k, v in substitution_map.items():
            G.add_edge(k, v)

        # check the integrity of the substitutions
        for node in G.nodes(data=True):
            if node[1]['bipartite'] == 0:  # term to be replaced node
                for neighbor in G.successors(node[0]):
                    if G.nodes[neighbor]['bipartite'] == 0:  # successor is also a term to be replaced
                        raise ValueError(f'Invalid `substitution_map`! The term "{neighbor}" cannot be both replaced and a replacement!')
            else:  # replacement term node
                if G.out_degree(node[0]) > 0:
                    if set(G.successors(node[0])) != {node[0]}:
                        raise ValueError(f'Invalid `substitution_map`! The term "{node[0]}" cannot be both replaced and a replacement!')
        return True
    
    
    def _permute_substitutions(self, substitution_map):
        """
        Permute substitution_map keys to account for tokenization differences (spaces and hyphens)

        For each key in the input dictionary, this function generates permutations of the key 
        by considering possible substitutions using the characters '-' and ' '. The resulting 
        dictionary will have expanded keys, where each original key-value pair is mapped to 
        multiple new key-value pairs based on the permutations of the original key.

        Parameters:
        -----------
        substitution_map: dict
            The substitution map

        Returns:
        --------
        dict: 
            Permuted substitution map
        """
        new_substitution_map = {}
        for k, v in substitution_map.items():
            for permuted_k in self._permute_substitutions_helper(k, '- '):
                new_substitution_map[permuted_k] = v
        return new_substitution_map
    
    
    def _permute_substitutions_helper(self, text, splitters):
        """
        Generate all possible permutations of a given text by substituting the characters in 'splitters'.

        This method finds all occurrences of splitter characters in the given text and then generates
        all possible combinations of those splitters for their respective positions. 

        Parameters:
        -----------
        text: str
            The input text for which permutations are to be generated.
        splitters: list, str
            A list/string containing characters that should be considered for permutation.

        Returns:
        --------
        list: 
            A list containing all possible permutations of the input text with the given splitters.

        Example:
        --------
        >>> _permute_substitutions_helper("a-b c", "- ")
        ['a-b c', 'a b c', 'a-b-c', 'a b-c']
        """
        # find all occurrences of splitters
        splitter_positions = [i for i, char in enumerate(text) if char in splitters]

        # generate all combinations of splitters for given positions
        splitter_combinations = itertools.product(list(splitters), repeat=len(splitter_positions))

        permutations = []
        for combination in splitter_combinations:
            temp_text = list(text)
            for index, splitter in zip(splitter_positions, combination):
                temp_text[index] = splitter
            permutations.append("".join(temp_text))
        return permutations
    
    
    def _lower_substitutions(self, substitution_map):
        """
        Convert the keys of the substitution map to lowercase. If two keys are equivalent in 
        lowercase and have different values, an error is raised.

        Parameters:
        -----------
        substitution_map: dict
            The substitution map

        Returns:
        --------
        dict: 
            New substitution map with all keys in lowercase.

        Raises:
        -------
        ValueError: 
            If two keys in the input dictionary are equivalent in lowercase 
            and have different values.
        """
        temp_substitution_map = {}
        for key, value in substitution_map.items():
            lowercase_key = key.lower()
            if lowercase_key in temp_substitution_map:
                if temp_substitution_map[lowercase_key] != value:
                    raise ValueError(f'Keys "{key}" and ' \
                                     f'"{[k for k, v in substitution_map.items() if k.lower() == lowercase_key and v != value][0]}" ' \
                                      'are equivalent in lowercase but have different values!')
            else:
                temp_substitution_map[lowercase_key] = value
        return temp_substitution_map
    
    
    def _lemmatize_substitutions(self, substitution_map):
        """
        Lemmatize the keys of the substitution map. If two keys are equivalent post 
        lemmatization and have different values, an error is raised.

        Parameters:
        -----------
        substitution_map: dict
            The substitution map

        Returns:
        --------
        dict: 
            New substitution map with added lemmatized keys

        Raises:
        -------
        ValueError: 
            If two keys in the input dictionary are equivalent post lemmatization 
            and have different values.
        """
        temp_substitution_map = {}
        cleaner = LemmatizeCleaner(self.lemmatize)
        for i, (key, value) in enumerate(substitution_map.items()):
            _, lemmatized_key = cleaner((i, key))
            if lemmatized_key in temp_substitution_map:
                if temp_substitution_map[lemmatized_key] != value:
                    raise ValueError(f'Keys "{key}" and ' \
                                     f'"{[k for k, v in substitution_map.items() if cleaner((_, k))[1] == lemmatized_key and v != value][0]}"' \
                                      ' are equivalent in lemmatized form but have different values!')
            else:
                temp_substitution_map[lemmatized_key] = value
        return temp_substitution_map
    
    
    # GETTERS / SETTERS

    
    @property
    def substitution_map(self):
        return self._substitution_map
    
    @substitution_map.setter
    def substitution_map(self, substitution_map):
        if substitution_map is None:
            warnings.warn('[SubstitutionCleaner]: No `substitution_map` was provided! ' \
                          'Ignoring substitution!', RuntimeWarning)
            self._substitution_map = {}
        elif isinstance(substitution_map, dict):
            substitution_map = substitution_map.copy()  # break reference to input argument
            is_valid = self._validate_substitutions(substitution_map)
            if is_valid:
                if self.lower:
                    substitution_map.update(self._lower_substitutions(substitution_map))
                if self.lemmatize:
                    substitution_map.update(self._lemmatize_substitutions(substitution_map))
                if self.permute:
                    substitution_map.update(self._permute_substitutions(substitution_map))
                self._substitution_map = dict(sorted(substitution_map.items(), \
                                                     key=lambda x: len(x[0].split()), \
                                                     reverse=True))
                
                # add substitutions to frozen
                for target in self.substitution_map.values():
                    self.frozen.add(target)
            else:
                raise ValueError('All keys and values in `substitution_map` must be strings!')
        else:
            raise TypeError('`substitution_map` must be a dict!')
            
    def _set_truth_value(self, value, name):
        try:
            return bool(value)
        except Exception as e:
            raise TypeError(f'Cannot extract truth value for `{name}` from "{type(permute)}"!')    
    
    @property
    def permute(self):
        return self._permute
    
    @permute.setter
    def permute(self, permute):
        self._permute = self._set_truth_value(permute, 'permute')
        
    @property
    def lower(self):
        return self._lower
    
    @lower.setter
    def lower(self, lower):
        self._lower = self._set_truth_value(lower, 'lower')
        
    @property
    def lemmatize(self):
        return self._lemmatize
    
    @lemmatize.setter
    def lemmatize(self, lemmatize):        
        if not isinstance(lemmatize, (str, bool)):
            raise TypeError('Expected type str or bool for `lemmatize`!')
        if isinstance(lemmatize, bool):
            if lemmatize:
                self._lemmatize = LemmatizeCleaner.BACKEND_LIBRARIES[0]
            else:
                self._lemmatize = '' # empty string means lemmatize is disabled
        else:
            self._lemmatize = lemmatize