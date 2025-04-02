import re
import nltk
import spacy
import warnings

from .module_base import VultureModuleBase


class LemmatizeCleaner(VultureModuleBase):
    """
    A text cleaner that normalizes tokens through lemmatization.
    
    Lemmatization is a process where words are reduced to their base or dictionary form, 
    known as the lemma. Unlike stemming, which crudely chops off word endings, lemmatization 
    can consider the context and part of speech of a word in a sentence to convert it to 
    its base form, which has a real linguistic root. For example, the lemma of 'running' is 'run',
    and the lemma of 'mice' is 'mouse'. This process helps in standardizing words to their 
    root form, normalizing the semantics of the text. This class implements lemmatization 
    in the Vulture pipeline and supports multiple external lemmatization backends.
    
    Attributes:
    -----------
    library: str 
        The name of the library that is used for the lemmatization backend 
    """
    # supported lemmatization libraries
    BACKEND_LIBRARIES = ['spacy', 'nltk']
    
    def __init__(self, library, frozen=None):
        super().__init__(frozen)
        self.module_type = "CLEANER"
        self.library = library
        self.backend = None
        
    def __call__(self, document):
        return self.run(document)
        
        
    def run(self, document):
        """
        Run the lemmatization

        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to perform lemmatization

        Returns
        -------
        tuple
            Tuple of document id and substituted text
        """
        doc_id, doc_text = document
        doc_text = self._lemmatize(doc_text)
        return (doc_id, doc_text)
    
    
    def _lemmatize(self, text):
        """
        Lemmatize a given string

        Parameters
        ----------
        text: str
            A string to be lemmatized

        Returns
        -------
        str
            The lemmatized string
        """
        if self.backend is None:
            self._init_backend()
        
        lemmatized_tokens = []
        if self.library == 'nltk':
            for t in text.split():
                lemmatized_tokens.append(self.backend.lemmatize(t) if t not in self.frozen else t)
        
        elif self.library == 'spacy':
            placeholders = {}  # preprocess to replace frozen tokens with placeholders
            for token in self.frozen:
                if '-' in token:
                    placeholder = f"~~{token.replace('-', '_')}~~"
                    placeholders[placeholder] = token
                    text = text.replace(token, placeholder)

            lemma_text = self.backend(text)
            for t in lemma_text:
                if t.text in placeholders:
                    lemmatized_tokens.append(placeholders[t.text])
                else:
                    lemmatized_tokens.append(t.lemma_ if t.text not in self.frozen else t.text)
        else:
            warnings.warn(f'[LemmatizeCleaner]: Using unknown library "{self.library}"')
            lemmatized_tokens = text.split()  # default to no lemmatization
        
        # return string where hyphens are not split
        return ' '.join(lemmatized_tokens).replace(' - ', '-')
    
    
    def _init_backend(self):
        """
        Change lemmatization backend depending on library
        """
        if self.library == 'nltk':
            self.backend = nltk.stem.WordNetLemmatizer()
        elif self.library == 'spacy':
            self.backend = spacy.load('en_core_web_lg', disable=['ner', 'parser'])
        else:
            warnings.warn(f'[LemmatizeCleaner]: Using unknown library "{self.library}"') 
            self.backend = None
        
    
    # GETTERS / SETTERS

    
    @property
    def library(self):
        return self._library
    
    @library.setter
    def library(self, library):
        if not isinstance(library, str):
            raise TypeError('Expected type str for `library`!')
        if library not in self.BACKEND_LIBRARIES:
            raise ValueError(f'Unknown library "{library}"! Supported options are {self.BACKEND_LIBRARIES}.')
        self._library = library