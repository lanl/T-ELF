import re
import warnings
import unicodedata
from .module_base import VultureModuleBase


class SimpleCleaner(VultureModuleBase):
    """
    A text cleaner that can use simple text processing and regular expressions to clean the text. 
    
    A collection of standard regular expressions is pre-defined in the class. Alternatively, 
    the user has the ability to specify custom regular expressions for specific datasets.
    
    Attributes:
    -----------
    """
    # collection of all unicode characters that represent a hyphen ('-')
    HYPHENS = r'[\u002D\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2E3A\u2E3B]'
    
    # pre-defined common regular expressions for cleaning
    # useful in the general case, more specific datasets may require custom expressions
    DEFAULT_PATTERNS = {
        
        # replace all possible characters used for hypens with a single standard 
        "standardize_hyphens": (re.compile(HYPHENS), '-'),
        
        # try removing copyright statements with copyright symbol as reference
        "remove_copyright_statement": None,
        
        # remove phrases using regex patterns
        "remove_stop_phrases": None,
        
        # convert the text to lowercase (preserving frozen)
        "make_lower_case": None,
        
        # convert non standard text to a normalized version
        "normalize": None,
        
        # remove dashes not surrounded by alphanumeric characters
        'remove_trailing_dash': (r'(?<!\w)-|-(?!\w)', ''),
        
        # make hyphens single word
        'make_hyphens_words': (r"([a-z])\-([a-z])", ''),
        
        # remove next line characters
        'remove_next_line': (r'\n+', ' '),
        
        # remove emails
        'remove_email': (r'\S*@\S*\s?', ''),
        
        # remove any string that has '=', '/', '\', '±', or '≈' inside
        'remove_formulas': (r'\b\w*[\=\≈\/\\\±]\w*\b', ''),
        
        # replace dashes with spaces
        'remove_dash': (r'-', ''),
        
        # remove everything in between []
        'remove_between_[]': (r'\[.*?\]', ' '),
        
        # remove everything in between ()
        'remove_between_()': (r'\(.*?\)', ' '),
        
        # replace [ and ] with space
        'remove_[]': (r'[\[\]]', ' '),
        
        # replace ( and ) with space
        'remove_()': (r'[()]', ' '),  
        
        # replace \ with space
        'remove_\\': (r'\\', ' '),

        # remove numbers
        'remove_numbers': (r'\d+', ''),
        
        # remove numbers between word boundaries
        'remove_standalone_numbers': (r'\b\d+\b', ''),
        
        # remove non ASCII characters
        'remove_nonASCII_boundary': (r'\b[^\x00-\x7F]+\b', ''),
        
        # remove non ASCII characters
        'remove_nonASCII': (r'[^\x00-\x7F]+', ''),
        
        # remove tags
        'remove_tags': (r'&lt;/?.*?&gt;', ''),
        
        # remove any other special characters missed above
        'remove_special_characters': (r'[!|"|#|$|%|&|\|\'|(|)|*|+|,|.|/|:|;|<|=|>|?|@|[|\|]|^|_|`|{|\||}|~]', ''),  
        
        # break up 'frozen' terms that are surrounded by whitespace
        'isolate_frozen': None,
        
        # remove any extra whitespace
        'remove_extra_whitespace': (r'\s+', ' '),  
        
        # remove stop words
        'remove_stop_words': None,
        
        # keep tokens that have at least this many characters
        'min_characters': None,
    }
    

    def __init__(self, 
                 custom_patterns=None, 
                 stop_words=None, 
                 stop_phrases=None, 
                 min_characters=2, 
                 exclude_hyphenated_stopwords=False,
                 order=None, 
                 frozen=None,
                 ):
        
        self._frozen = set()
        self.module_type = "CLEANER"
        self.effective_stop_words = None
        self.patterns = self.DEFAULT_PATTERNS.copy()
        self.custom_patterns = custom_patterns
        if self.custom_patterns:
            self.patterns.update(self.custom_patterns)
        
        self.stop_words = stop_words
        self.stop_phrases = stop_phrases
        self.min_characters = min_characters
        self.order = order
        self.exclude_hyphenated_stopwords = exclude_hyphenated_stopwords
        
        self.sw_pattern = re.compile(r'\b[\w-]+\b')
        super().__init__(frozen)
        
        
    def __call__(self, document):
        return self.run(document)
        
        
    def run(self, document):
        """
        Run the simple cleaning

        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to check if english

        Returns
        -------
        tuple
            (Document ID, cleaned document text)
        """
        doc_id, doc_text = document
        cleaned_doc_text = self._apply_patterns(doc_text)
        return (doc_id, cleaned_doc_text)
        
        
    def _apply_patterns(self, text):
        """Apply regex patterns to the text."""
        for k in self.order:
            item = self.patterns[k]
            if item is None:
                if k == 'remove_copyright_statement':
                    text = self._remove_copyright(text)
                elif k == 'remove_stop_phrases':
                    text = self._remove_stop_phrases(text)
                elif k == 'make_lower_case':
                    text = self._lower(text)
                elif k == 'remove_stop_words':
                    text = self._remove_stop_words(text)
                elif k == 'normalize':
                    text = self._normalize(text)
                elif k == 'min_characters':
                    text = self._remove_short_tokens(text)
                elif k == 'isolate_frozen':
                    text = self._isolate_frozen(text)
                else:
                    pass
            else:
                pattern, replacement = item
                if not self.frozen:
                    text = re.sub(pattern, replacement, text)
                else:
                    matches = list(re.finditer(pattern, text))
                    
                    # process each match in reverse so replacing doesnt affect positions of future matches
                    for match in reversed(matches):
                        
                        is_frozen = False
                        start, end = match.span()
                        for word in self.frozen:
                            for m in re.finditer(re.escape(word), text):
                                if m.start() <= start < m.end():
                                    is_frozen = True
                                    break
                            if is_frozen:
                                break

                        if not is_frozen:
                            text = text[:start] + replacement + text[end:]
                            
        return text.strip()
    
    
    def _lower(self, text):
        """
        Convert text to lowercase. Preserve case for frozen tokens (words seen in self.frozen).

        Parameters
        ----------
        text: str
            Document text.

        Returns
        -------
        str
            Processed document text.
        """
        if not self.frozen:
            return text.lower()
        else:
            tokens = text.split()
            return ' '.join([t if t in self.frozen else t.lower() for t in tokens])
  

    def _remove_stop_words(self, text ):
        """
        Removes stop words from text

        Parameters
        ----------
        text: str
            Document text.

        Returns
        -------
        str
            Processed document text.
        """
        if not self.stop_words:
            warnings.warn('[SimpleCleaner]: Requested stop word removal but no '
                          'stop words were provided!', RuntimeWarning)
            return text
        else:
            tokens = self.sw_pattern.findall(text)

            if self.exclude_hyphenated_stopwords:
                cleaned_words = [t for t in tokens if 
                                t in self.frozen or  # entire term in frozen
                                not t.lower() in self.effective_stop_words] 
                return ' '.join(cleaned_words)
            else:
                cleaned_words = [t for t in tokens if 
                                t in self.frozen or  # entire term in frozen
                                not any(part.lower() in self.effective_stop_words for part in t.split('-'))]  # no part in stopwords
                return ' '.join(cleaned_words)

        
    def _remove_stop_words_helper(self, text):
        for s in self.frozen:
            if text in s:
                print(text, s)
                return True
        return False
        
        
    def _remove_stop_phrases(self, text):
        """
        Removes stop phrases from text

        Parameters
        ----------
        text: str
            Document text.

        Returns
        -------
        str
            Processed document text.
        """
        if not self.stop_phrases:
            warnings.warn('[SimpleCleaner]: Requested stop phrase removal but no '
                          'stop phrases were provided!', RuntimeWarning)
        for phrase in self.stop_phrases:
            text = re.sub(phrase, "", text, flags=re.IGNORECASE)
        return text


    def _remove_copyright(self, text):
        """
        Removes copyright statement from text

        Parameters
        ----------
        text: str
            Document text.

        Returns
        -------
        str
            Processed document text.
        """
        head, sep, tail = text.rpartition('©')
        if len(head) > len(tail):  # copyright at the end of abstract
            return head
        else:
            if not head and sep:  # copyright as first symbol
                try:
                    return text.split('.', 1)[1]
                except IndexError:
                    return ''  # all document is a copyright
            else:
                return text  # copyright not found
            
            
    def _normalize(self, text):
        """
        Normalize the given text by removing accented characters.

        The function decomposes the text into its canonical decomposition using the 'NFD' method.
        It then filters out characters that belong to the "Nonspacing Mark" category (e.g., diacritics)
        and returns the normalized string.

        Parameters:
        -----------
        text: str
            The input text to be normalized.

        Returns:
        --------
        str:
            The normalized text.
        """
        category = unicodedata.category
        result = [c for c in unicodedata.normalize('NFD', text) if category(c) != 'Mn']
        return ''.join(result)

    
    def _remove_short_tokens(self, text):
        """
        Removes tokens from the text that are shorter than the specified minimum length.

        Parameters
        ----------
        text: str
            The input text.
            
        Returns
        -------
        str
            The processed text with short tokens removed.
        """
        tokens = text.split() 
        return ' '.join([t for t in tokens if len(t) >= self.min_characters or t in self.frozen])

    
    def _isolate_frozen(self, text):
        """
        Search for occurrences of frozen strings in the given text and replaces
        non-letter characters immediately preceding or following each occurrence with a whitespace.

        Parameters
        ----------
        text: str
            The input text.
            
        Returns
        -------
        str
            The processed text with short tokens removed.
        """
        for target in self.frozen:
            text_list = list(text)
            found = [i for i in range(len(text)) if text.startswith(target, i)]

            for index in found:
                if index - 1 >= 0 and not text_list[index - 1].isalpha():  # preceeding character
                    text_list[index - 1] = ' '  # following character
                if index + len(target) < len(text) and not text_list[index + len(target)].isalpha():
                    text_list[index + len(target)] = ' '
            
            text = ''.join(text_list)
        return text

    
    # GETTERS / SETTERS
    
    
    def _validate_order(self, item, name):
        """Validate the order of regex patterns"""
        if item is None:
            return list(self.patterns.keys())
        if isinstance(item, (list, set)):
            order = set(item)
            if not order.issubset(set(self.patterns.keys())):
                raise ValueError(f'`{name}` contains unknown/undefined patterns!')
            return list(item)
        raise TypeError(f'`{name}` must be a list/set of strings!')
    
    
    def _validate_str_set(self, item, name):
        """Validate that item is a list or set of unique strings"""
        if item is None:
            return set()
        if isinstance(item, (list, set)):
            if all(isinstance(i, str) for i in item):  # check all elements are strings
                set_item = set(item)
                if len(set_item) < len(item):
                    warnings.warn(f'[SimpleCleaner]: `{name}` contains duplicate entries!', RuntimeWarning)
                return set_item
            else:
                raise TypeError(f'`{name}` must be a list of strings!')
        raise TypeError(f'`{name}` must be a list of strings!')
    

    def _validate_custom_patterns(self, item, name):
        """Validate that item is a dictionary for custom regex patterns"""
        if item is None:
            return {}
        if not isinstance(item, dict):
            raise TypeError(f'`{name}` must be a dictionary!')
        for key, value in item.items():
            if not isinstance(key, str):
                raise TypeError(f'`{name}` must contain strings as keys not "{type(key)}"!')
            if not isinstance(value, tuple) or not all(isinstance(i, str) for i in value):
                raise TypeError(f'`{name}` must contain tuple of (regular_expression, '\
                                f'replacement_value) for dict values!')

            # check if regex pattern is valid
            try:
                re.compile(value[0])
            except re.error:
                raise ValueError(f'"{value[0]}" is not a valid regular expression')
        return item
    
    @property
    def custom_patterns(self):
        return self._custom_patterns

    @custom_patterns.setter
    def custom_patterns(self, item):
        self._custom_patterns = self._validate_custom_patterns(item, "custom_patterns")
    
    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, item):
        self._stop_words = self._validate_str_set(item, "stop_words")
        self._update_effective_stop_words()
        
    @property
    def stop_phrases(self):
        return self._stop_phrases

    @stop_phrases.setter
    def stop_phrases(self, item):
        self._stop_phrases = self._validate_str_set(item, "stop_phrases")
    
    @property
    def min_characters(self):
        return self._min_characters
    
    @min_characters.setter
    def min_characters(self, item):
        if not isinstance(item, int):
            raise TypeError('`min_characters` must be an int')
        if item < 0:
            raise ValueError('`min_characters must be a positive int')
        self._min_characters = item
    
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, item):
        self._order = self._validate_order(item, "order")
        
    @property
    def frozen(self):
        return self._frozen

    @VultureModuleBase.frozen.setter
    def frozen(self, item):
        VultureModuleBase.frozen.fset(self, item)
        self._update_effective_stop_words()
        
    def _update_effective_stop_words(self):
        effective_stop_words = [word for word in self._stop_words if word not in self._frozen]
        if len(effective_stop_words) < len(self._stop_words):
            warnings.warn('[SimpleCleaner]: One or more stopwords are frozen from cleaning ' \
                          'due to substitution!', RuntimeWarning)
        self.effective_stop_words = sorted([re.escape(word) for word in effective_stop_words],  key=len, reverse=True)