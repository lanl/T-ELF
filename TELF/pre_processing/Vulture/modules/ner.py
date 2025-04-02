import re
import spacy
import warnings

from .module_base import VultureModuleBase

class NEDetector(VultureModuleBase):
    """
    An operator that detects Named Entities in text.
    
    Attributes:
    -----------
    library: str 
        The name of the library that is used for the NER backend 
    """
    # supported lemmatization libraries
    BACKEND_LIBRARIES = ['en_core_web_trf', 'en_core_web_lg']
    
    def __init__(self, library="en_core_web_trf", frozen=None):
        super().__init__(frozen)
        self.module_type = "OPERATOR"
        self.library = library
        self.backend = None
        
    def __call__(self, document):
        return self.run(document)
        
        
    def run(self, document):
        """
        Run the NER detection

        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to perform NER detection

        Returns
        -------
        tuple
            Tuple of document id and operation result
        """
        doc_id, doc_text = document
        doc_operation_result = self._detect_NER(doc_text)
        return (doc_id, doc_operation_result)
    
    
    def _detect_NER(self, text):
        """
        Detect NERs in a given string

        Parameters
        ----------
        text: str
            A string to be NER detection performed on

        Returns
        -------
        str
            Dictionary of entity name and correcponding set of entities
        """
        if self.backend is None:
            self._init_backend()

        doc = self.backend(text)
        entities_set = set()  
        label_entities = {}
        for ent in doc.ents:
            entities_set.add((ent.text, ent.label_))
            if ent.label_ not in label_entities:
                label_entities[ent.label_] = set()
            label_entities[ent.label_].add(ent.text)

        # return string where hyphens are not split
        return label_entities
    
    
    def _init_backend(self):
        """
        Change NER detection backend depending on library
        """
        self.backend = spacy.load(self.library)
        
    
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