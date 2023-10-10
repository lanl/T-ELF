import re

COMMON_STOPWORDS = {'the','to','a','of','and','in','that','for','is','on','said','with','he','it','was','as','at','but','by','his','be','have','are','from','i','has','not','an','they','who','this','or','had','you','their','about','will','one','more','when','we','would','new','were','been','if','she','can','out','all','up','which','her','its','there','than','what','some','like','no','so','after','two','other','its','into','also','just','them','him','most','over','do','could','only','now','because','many','even','my','get','before','how','then','where','those','much','dont','any','may','here','still','made','good','through','did','our','while','going','off','me','these','think','against','such','since','down','very','another','being','go','should','too','under','both','during','between'}


def ascii_detect(text):
    """ Find the total number of ASCII characters in the dataset
    
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


def simple_tokenize(text):
    """ Tokenize text without doing any serious pre-processing.
    
    Parameters
    ----------
    text : str
        String that needs to be tokenized
    
    Returns
    -------
    tokens : list(str)
       The list of tokens from text
    """
    text = text.lower()  # Convert the string to lowercase
    text = re.sub(r'[^a-z0-9\s-]+', '', text)  # Remove all non-alphanumeric characters, except hyphens
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra white spaces
    return text.split()


def is_english(document: str, ascii_ratio=0.9, stopwords_used=0.3):
    """
    Performs language detection on the given documents

    Parameters
    ----------
    document : str
        The document text for which to check if english
    
    Returns
    -------
    is_english : bool
        Is the document English (according to this heuristic)
    """
    ascii_count, total_count = ascii_detect(document)
    if ascii_count / total_count < ascii_ratio:
        return False
    
    tokens = simple_tokenize(document)
    num_stopwords = sum(1 for t in tokens if t in COMMON_STOPWORDS)
    if num_stopwords / len(tokens) < stopwords_used:
        return False
        
    return True
