import re
from TELF.pre_processing.Vulture.modules import VultureModuleBase

class SubstitutionOperator(VultureModuleBase):
    """
    An operator that performs per document substitutions.

    """

    def __init__(self, 
                 document_substitutions = None,
                 corpus_substitutions = None,
                 document_priority = True,
                 frozen=None):
        
        super().__init__(frozen)
        self.module_type = "OPERATOR"
        self.current_document_id = None
        self.document_substitutions = document_substitutions
        self.corpus_substitutions = corpus_substitutions
        self.document_priority = document_priority

    def __call__(self, document):
        return self.run(document)
        
    def run(self, document):
        """
        Runs the substitutions per document with the two substitutions.
        
        Parameters
        ----------
        document: tuple
            A document id, document text pair for which to perform acronym detection

        Returns
        -------
        tuple
            Tuple of document id and operation result
        """
        doc_id, doc_text = document
        self.current_document_id = doc_id
    
        if self.corpus_substitutions and self.document_substitutions:
            current_document_subs = self.document_substitutions[doc_id] 
            substitution_map = {**self.corpus_substitutions, **current_document_subs} if self.document_priority else {**current_document_subs, **self.corpus_substitutions}
        elif self.corpus_substitutions:
            substitution_map =  self.corpus_substitutions
        elif self.document_substitutions:
            substitution_map = self.document_substitutions
        else:
            raise ValueError("No substitutions were passed to the SubstitutionOperator.")
        
        doc_operation_result = self._document_substitution(doc_text, substitution_map)
        return (doc_id, doc_operation_result)
    

    def  _document_substitution(self, text, substitution_map):
        """

        Parameters 
        ----------
        text: str
            A string to replace text in

        Returns
        -------
        str
            Dictionary of entity name and corresponding set of entities
        """
        replaced_text = text
        for search, replace in substitution_map.items():
            replaced_text = re.sub(r'\b{}\b'.format(re.escape(search)), replace, replaced_text)                
        return {"replaced_text":replaced_text}
    