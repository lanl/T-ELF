import re
from TELF.pre_processing.Vulture.modules import VultureModuleBase


class RemoveNonEnglishCleaner(VultureModuleBase):
    """
    A text cleaner that detects if the text is English 
    
    The English detection is done through a simple heuristic. This heuristic uses two
    components: the ratio of ASCII characters to all characters and the ratio of stopwords
    to all words. While not flawless, this heuristic (with appropriate hyperparameters)
    provides a quick way to filter out foreign language documents.
    
    Attributes:
    -----------
    ascii_ratio: float 
        The minimum acceptable ratio of ASCII characters to all characters
    stopword_ratio: float
        The minimum acceptable ratio of stopwords to all words
    """
    
    # A sample of the most common stopwords used in the English language
    # Derived from https://faculty.georgetown.edu/wilsong/IR/WD3.html
    COMMON_STOPWORDS = {'the','to','a','of','and','in','that','for','is','on','said','with'
                        'he','it','was','as','at','but','by','his','be','have','are','from',
                        'i','has','not','an','they','who','this','or','had','you','their',
                        'about','will','one','more','when','we','would','new','were','been',
                        'if','she','can','out','all','up','which','her','its','there','than',
                        'what','some','like','no','so','after','two','other','its','into',
                        'also','just','them','him','most','over','do','could','only','now',
                        'because','many','even','my','get','before','how','then','where',
                        'those','much','dont','any','may','here','still','made','good','through',
                        'did','our','while','going','off','me','these','think','against','such',
                        'since','down','very','another','being','go','should','too','under',
                        'both','during','between'}

    
    def __init__(self, ascii_ratio=0.9, stopwords_ratio=0.2, frozen=None):
        super().__init__(frozen)  # initialize the base class with the preserve
        self.module_type = "CLEANER"
        self.ascii_ratio = ascii_ratio
        self.stopwords_ratio = stopwords_ratio
        
        
    def __call__(self, document):
        return self.run(document)
        
        
    def run(self, document):
        """
        Run the non-english language detection heuristic

        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to check if english

        Returns
        -------
        bool
            Is the document English (according to this heuristic)
        """
        doc_id, doc_text = document
        ascii_count, total_count = self._ascii_detect(doc_text)
        try:
            if ascii_count / total_count < self.ascii_ratio:
                return (doc_id, '')
        except ZeroDivisionError:
            return (doc_id, '')

        tokens = self._simple_tokenize(doc_text)
        num_stopwords = sum(1 for t in tokens if t in self.COMMON_STOPWORDS)
        
        try:
            if num_stopwords / len(tokens) < self.stopwords_ratio:
                return (doc_id, '')
        except ZeroDivisionError:
            return (doc_id, '')
        
        return document
    
    
    def _ascii_detect(self, text):
        """ 
        Find the total number of ASCII characters in the dataset

        Parameters
        ----------
        text : str
            String for which to find the number of ascii characters

        Returns
        -------
        ascii_count : int
            Number of ascii characters in text
        total_count : int
            Total number of characters in text (excluding whitespace)
        """
        text_no_whitespace = ''.join(text.split())  # remove whitespace
        ascii_count = sum(1 for c in text_no_whitespace if ord(c) < 128)  # count ASCII characters
        total_count = len(text_no_whitespace)  # count total characters
        return ascii_count, total_count


    def _simple_tokenize(self, text):
        """ 
        Tokenize text without doing any serious pre-processing.

        Parameters
        ----------
        text : str
            String that needs to be tokenized

        Returns
        -------
        tokens : list(str)
           The list of tokens from text
        """
        text = text.lower()  # convert the string to lowercase
        text = re.sub(r'[^a-z0-9\s-]+', '', text)  # remove all non-alphanumeric characters, except hyphens
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra white spaces
        return text.split()

    
    # GETTERS / SETTERS
    
    
    def _validate_ratio(self, value, name):
        """Validate that the ratio is numeric and between 0 and 1."""
        if not isinstance(value, (int, float)):
            raise TypeError(f'{name} must be numeric!')
        if not (0 < value < 1):
            raise ValueError(f'{name} must be greater than 0 and less than 1!')
        return value

    @property
    def ascii_ratio(self):
        return self._ascii_ratio

    @ascii_ratio.setter
    def ascii_ratio(self, value):
        self._ascii_ratio = self._validate_ratio(value, "ascii_ratio")

    @property
    def stopwords_ratio(self):
        return self._stopwords_ratio

    @stopwords_ratio.setter
    def stopwords_ratio(self, value):
        self._stopwords_ratio = self._validate_ratio(value, "stopwords_ratio")