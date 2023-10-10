from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
import spacy
import re
import warnings

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import defaultdict
from collections import Counter

def stem_document(document: str, document_id: str) -> tuple:
    """
    Performs stemming of a given document.

    Parameters
    ----------
    document : str
        single text document.
    document_id : str
        unique identifier of the given document.

    Returns
    -------
    tuple
        document ID, stemmed document pair.

    """
    tokens = document.split()
    stemmer = SnowballStemmer("english")
    stem_toks = [stemmer.stem(word=word) for word in tokens]
    stem_document = " ".join(stem_toks)
    return (document_id, stem_document)


def lemmatize_document(document: str, document_id: str) -> tuple:
    """
    Performs lemmatization of a given document.

    Parameters
    ----------
    document : str
        single text document.
    document_id : str
        unique identifier of the given document.

    Returns
    -------
    tuple
        document ID, lemmatized document pair.

    """
    tokens = document.split()
    lemmatizer = WordNetLemmatizer()
    lemma_toks = [lemmatizer.lemmatize(word=word, pos=wordnet.VERB) for word in tokens]
    lemma_document = " ".join(lemma_toks)

    return (document_id, lemma_document)


def advance_document_clean(
        document: str,
        document_id: str,
        nlp,
        allowed_postags={'NOUN': 1, 'ADJ': 1, 'VERB': 1, 'ADV': 1, "PROPN": 1},
        lemmatize=True,
) -> tuple:
    """
    Form Bigrams, Trigrams, filter to the allowed postags and apply spacy lemmatization

    Parameters
    ----------
    document : str
        single text document.
    document_id : str
        unique identifier of the given document.
    nlp : callable
        Spacy NLP model.
    allowed_postags : list, optional
        The list of allowed Postags. The default is ['NOUN', 'ADJ', 'VERB', 'ADV', "PROPN"].
    lemmatize : bool, optional
        Parameter to select if we are doing lemmatization

    Returns
    -------
    tuple
        document ID, cleaned document pair.

    """
    # make list of words from document
    tokens = document.split()

    # make the document from the words
    document = " ".join(tokens)

    # apply Spacy postag cleaning
    nlp_doc = nlp(document)
    document_filtered = []

    if lemmatize:
        document_filtered = [token.lemma_ for token in nlp_doc if token.pos_ in allowed_postags]
    else:
        document_filtered = [str(token) for token in nlp_doc if token.pos_ in allowed_postags]

    # put together the tokens
    document_filtered = " ".join(document_filtered)

    return(document_id, document_filtered)


def check_character_lengths(
        document: str,
        document_id: str,
        min_characters=2,
        min_unique_characters=2) -> tuple:
    """
    check tokens for character lengths


    Parameters
    ----------
    document : str
        single text document.
    document_id : str
        unique identifier of the given document.
    min_characters : int, optional
        Minimum number of characters a token should have. The default is 2.
    min_unique_characters : int, optional
        Minimum number of unique characters a token should have. The default is 2.

    Returns
    -------
    tuple
        document ID, cleaned document pair.

    """

    tokens_filtered = []
    for token in document.split():

        # check for character counts
        if (len(token) >= min_characters) and len(set(token)) > min_unique_characters:
            tokens_filtered.append(token)

    # document from the tokens
    document_filtered = " ".join(tokens_filtered)

    return (document_id, document_filtered)


def simple_document_clean(
    document: str,
    document_id: str,
    stop_words,
    stop_phrases: list,
    clean_settings={}
) -> tuple:
    """
    Cleans the given document.

    Have the capability to do:
    - make lower case
    - remove copyright statements with copyright symbol
    - remove stop-phrases
    - make hypen words single word
    - remove next line characters
    - remove emails
    - replace dashes with spaces
    - replace [ and ] with space
    - replace \ with space
    - replace ^ with space
    - Remove numbers
    - remove non ASCII characters
    - remove tags
    - remove any other symbols
    - remove anything in between [], including []
    - filter stop-words

    Parameters
    ----------
    document : str
        single text document.
    document_id : str
        unique identifier of the given document.
    stop_words : dict (recommended), or list
        Hashmap (dict) of stopwords. O(1) lookup.
        List of stopwords. O(n) lookup.
    stop_phrases: list, optional
        List of phrases to be removed.
    clean_settings: dict, optional
        Settings used in simple cleaning. See _organize_simple_clean_defaults for defaults.

    Returns
    -------
    tuple
        document ID, cleaned document pair.

    """

    # collection of all unicode characters that represent a hyphen ('-')
    hyphen_chars = r'[\u002D\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2E3A\u2E3B]'
    
    if type(stop_words) != dict:
        warnings.warn("Using stop_words as list is significantly slower. \
         Use a hash-map (dict) instead for O(1) lookup.")

    # try removing copyright statements with copyright symbol as reference
    if clean_settings["remove_copyright_with_symbol"]:
        document = _remove_copyright(document)

    # remove phrases
    if clean_settings["remove_stop_phrases"]:
        document = _remove_stop_phrases(document, stop_phrases)

    # make lower case
    if clean_settings["make_lower_case"]:
        document = document.lower()

    # remove dashes not surrounded by alphanumeric characters
    if clean_settings["remove_trailing_dash"]:
        document = re.sub(hyphen_chars, '-', document)  # standardize hyphens
        document = re.sub(r'-+', '-', document)         # replace consecutive hyphens with a single hyphen
        pattern = r'(?<!\w)-|-(?!\w)'                   # remove hyphens not surrounded by alphanumeric characters
        document = re.sub(pattern, '', document)
        
    # make hypens single word
    if clean_settings["make_hyphens_words"]:
        document = re.sub(hyphen_chars, '-', document)  # standardize hyphens
        document = re.sub(r'-+', '-', document)         # replace consecutive hyphens with a single hyphen
        document = re.sub(r"([a-z])\-([a-z])", r"\1\2", document, 0, re.IGNORECASE)

    # remove next line characters
    if clean_settings["remove_next_line"]:
        document = document.replace("\n", " ").strip()
        document = re.sub("\s+", " ", document)

    # remove emails
    if clean_settings["remove_email"]:
        document = re.sub("\S*@\S*\s?", " ", document)
        
    # replace dashes with spaces
    if clean_settings["remove_dash"]:
        document = re.sub(hyphen_chars, '-', document)
        document = document.replace("-", " ")

    # remove everything in between []
    if clean_settings["remove_between_[]"]:
        document = re.sub("\[.*?\]", " ", document)
    
    # remove parentheses and any string contained within the parentheses
    if clean_settings["remove_between_()"]:
        document = re.sub(r"\(.*?\)", " ", document)

    # replace [ and ] with space
    if clean_settings["remove_[]"]:
        document = document.replace("[", " ")
        document = document.replace("]", " ")

    # replace ( and ) with space
    if clean_settings["remove_()"]:
        document = document.replace("(", " ")
        document = document.replace(")", " ")
        
    # replace \ with space
    if clean_settings["remove_\\"]:
        document = document.replace("\\", " ")

    # replace ^ with space
    if clean_settings["remove_^"]:
        document = document.replace("^", " ")

    # Remove numbers
    if clean_settings["remove_numbers"]:
        document = re.sub(r"\d+", "", document)

    # Remove numbers between word bounaries
    if clean_settings["remove_standalone_numbers"]:
        document = re.sub(r"\b\d+\b", "", document)

    # non ASCII characters
    if clean_settings["remove_nonASCII"]:
        document = re.sub(r"[^\x00-\x7F]+", " ", document)

    # remove tags
    if clean_settings["remove_tags"]:
        document = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", document)

    # remove any other special characters missed above
    if clean_settings["remove_special_characters"]:
        document = re.sub(
            r'[!|"|#|$|%|&|\|\'|(|)|*|+|,|.|/|:|;|<|=|>|?|@|[|\|]|^|_|`|{|\||}|~]', " ", document)

    # remove any extra whitespace
    document = re.sub(r'\s+', ' ', document).strip()
    
    # tokenize and remove whitespace if requested
    if clean_settings["remove_stop_words"]:
        tokens_filtered = [word for word in document.split() if word not in stop_words]
    else:
        tokens_filtered = document.split()

    # document from the tokens
    return (document_id, " ".join(tokens_filtered))


def _remove_stop_phrases(text: str, stop_phrases: list) -> str:
    """
    Removes phrases from text

    Parameters
    ----------
    text : str
        Document text.

    Returns
    -------
    text
        processed document text.

    """
    for phrase in stop_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)
    return text


def _remove_copyright(text: str) -> str:
    """
    Tries its best to remove copyright statement from string...

    Parameters
    ----------
    text : str
        Document text.

    Returns
    -------
    text
        processed document text.

    """
    head, sep, tail = text.rpartition("©")
    if len(head) > len(tail):  # copyright at the end of abstract
        return head
    else:
        if not head and sep:  # copyright as first symbol
            try:
                return text.split(".", 1)[1]
            except IndexError:
                return ""  # all document is a copyright
        else:
            return text  # copyright not found


def _organize_simple_clean_defaults(settings: dict) -> dict:
    """
    Organizes the default entries for the settings used in simple  cleaning.

    Parameters
    ----------
    settings : dict
        Dictionary of settings.

    Returns
    -------
    dict
        Default settings.

    """
    allowed_settings = ["make_hyphens_words", "remove_copyright_with_symbol", "remove_stop_phrases",
                        "make_lower_case", "remove_next_line", "remove_email", "remove_dash",
                        "remove_[]", "remove_()", "remove_\\", "remove_^", "remove_numbers", "remove_nonASCII",
                        "remove_tags", "remove_special_characters", "remove_between_[]", "remove_between_()",
                        "check_char_length", "remove_stop_words", "remove_trailing_dash", "remove_standalone_numbers"]

    for key, _ in settings.items():
        assert key in allowed_settings, f'Unknown setting {key} for simple cleaning. Choose from: {", ".join(allowed_settings)}'

    if "make_hyphens_words" not in settings:
        settings["make_hyphens_words"] = False

    if "remove_copyright_with_symbol" not in settings:
        settings["remove_copyright_with_symbol"] = False

    if "remove_stop_phrases" not in settings:
        settings["remove_stop_phrases"] = False

    if "make_lower_case" not in settings:
        settings["make_lower_case"] = True

    if "remove_next_line" not in settings:
        settings["remove_next_line"] = True

    if "remove_email" not in settings:
        settings["remove_email"] = True

    if "remove_dash" not in settings:
        settings["remove_dash"] = False

    if "remove_[]" not in settings:
        settings["remove_[]"] = True
        
    if "remove_()" not in settings:
        settings["remove_()"] = True

    if "remove_\\" not in settings:
        settings["remove_\\"] = True

    if "remove_^" not in settings:
        settings["remove_^"] = True

    if "remove_numbers" not in settings:
        settings["remove_numbers"] = True

    if "remove_standalone_numbers":
        settings["remove_standalone_numbers"] = False

    if "remove_nonASCII" not in settings:
        settings["remove_nonASCII"] = True

    if "remove_tags" not in settings:
        settings["remove_tags"] = True

    if "remove_special_characters" not in settings:
        settings["remove_special_characters"] = True

    if "remove_between_[]" not in settings:
        settings["remove_between_[]"] = True
        
    if "remove_between_()" not in settings:
        settings["remove_between_()"] = True

    if "check_char_length" not in settings:
        settings["check_char_length"] = True

    if "remove_stop_words" not in settings:
        settings["remove_stop_words"] = True
        
    if "remove_trailing_dash" not in settings:
        settings["remove_trailing_dash"] = True
    

    return settings


def strip_suffixes(word, suffixes ):
    """
    Removes found suffixes

    Parameters
    ----------
    word : str
        element to remove suffex from
    suffixes : list
        common suffixes in english
   
    Returns
    -------
    word :  str
        element without suffix 
    """
    for suffix in sorted(suffixes, key=len, reverse=True):  # sort by length to handle longer suffixes first
        if word.endswith(suffix):
            word = word[:-len(suffix)]
    return word

def nested_defaultdict():
    """
    can be called for another default dict, where the third dimension becomes a list 

    Parameters
    ----------
    None
   
    Returns
    -------
    default dict with list values
    """
    return defaultdict(Counter)


def build_stem_map(vocabulary, method = 'frequency'):
    """
    Stems vocabulary map, ununified

    Parameters
    ----------
    vocabulary : list
        Node containing a row of the data.
   
    Returns
    -------
    vocab_stems : dict (str:str)
        variants map to shortest variant, ununified
    """
    ps = PorterStemmer()
    vocab_stems = defaultdict( nested_defaultdict )

    for w in vocabulary:
        stem = ps.stem(w)
        vocab_stems[stem]['src'][w] += 1
        
        if method == 'shortest':
            destination_word = min(vocab_stems[stem]['src'], key=len)
        elif method == 'frequency':
            destination_word = vocab_stems[stem]['src'].most_common(1)[0][0]
        else:
            raise Exception("Need to call an existing method for slic stemmer: shortest or frquency")  
        vocab_stems[stem]['dest'] = destination_word
    import json

    return vocab_stems

 
def build_vocab_stem_subs(vocabulary: list, min_char_len: int = 4): #, suffixes: list):
    """
    Stems vocabulary, constructs map of all variants to the shorstest variant.

    Parameters
    ----------
    vocabulary : list
        Node containing a row of the data.
   
    Returns
    -------
    subs_stemed : dict (str:str)
        variants map to shortest variant
    shortened_vocabulary : list
        new vocab with only mapped variants
    """
    subs_stemed = {}
    vocab_stems = build_stem_map(vocabulary)
    
    # FOLLOWING LINE IS TEMPORARY -- NEED TO VERIFY VERACITY OF UNIFICATION PROCESS
    # vocab_stems = unify_common_stems(vocab_stems, suffixes)

    for stem in vocab_stems:
        for src in vocab_stems[stem]['src']:
            
            destination_word = vocab_stems[stem]['dest']
          
            # substitution map
            longer_than_3 = (len(destination_word) >= min_char_len) and (len(src) >= min_char_len)
            if src != destination_word and longer_than_3: # 
                subs_stemed[src] = destination_word
                
    return subs_stemed

def correct_text(documents: dict, corrections: dict):
    """
    Checks word boundaries for any and all occurances of corrections, replaces key with value

    Parameters
    ----------
    documents : dict
        id, text of corpus
    corrections : dict
        mapping of replacements, where key is source and value is destination 
   
    Returns
    -------
    results : list
        list of tuples containing the original id, text parings.
    """
    results = []
    for docID, text in documents.items():
        for search, replacement in corrections.items():
            text = re.sub(r'\b{}\b'.format(re.escape(search)), replacement, text)
        
        results.append((docID, text))
    return results
