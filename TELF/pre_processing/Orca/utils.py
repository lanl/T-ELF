import rapidfuzz
import unicodedata
from itertools import permutations


def generate_name_variations(name):
    """ 
    Generate a list of name variations for a given name. This function handles names
    with different structures, including those with and without middle names, and abbreviated
    names. The variations include different combinations of full names, initials, and
    reordering of name parts.
    
    Parameters:
    -----------
    name: str
        The name to be processed
    
    Returns:
    --------
    List of name variations
    
    Example:
    --------
    >>> generate_name_variations('John Randall Smith')
        ['John Smith', 'Randall S.J.', 'John Randall Smith', 'Smith John', 'John R. Smith', 
         'J.R. Smith', 'Randall J.S.', 'Smith J.R.', 'Smith John Randall']
    >>> generate_name_variations('John Smith')
        ['Smith John', 'John Smith']
    >>> generate_name_variations('Smith')
        ['Smith']
    >>> generate_name_variations('J.R. Smith')
        ['Smith J.R.', 'J.R. Smith']
    """    
    parts = name.split()  # split name into parts
    variations = [name]
    
    # only one part, assume it is either first or last name
    # nothing to do
    if len(parts) == 1:
        variations.append(parts[0])
        
    # name has multiple parts
    # process for normalization
    else:
        first_name = parts[0]
        last_name = parts[-1]

        # full name in normal order
        variations.append(' '.join(parts))
        
        if len(parts) > 2:
            middle_names = parts[1:-1]
            initials = "".join([n[0] + '.' for n in middle_names])

            # initials with last name
            variations.append(f"{first_name[0]}.{initials} {last_name}")

            # first name with middle initial and last name
            middle_initials = " ".join([n[0] + '.' for n in middle_names])
            variations.append(f"{first_name} {middle_initials} {last_name}")
            
            # last name followed by first and middle names
            variations.append(f"{last_name} {first_name} {' '.join(middle_names)}")
            
            # last name followed by initials
            variations.append(f"{last_name} {first_name[0]}.{initials}")

            # variations for the middle names
            for middle_name in middle_names:
                other_parts = [first_name[0], last_name[0]] + [n[0] for n in middle_names if n != middle_name]
                for perm in permutations(other_parts):
                    variations.append(f"{middle_name} {'.'.join([p for p in perm])}.")
            
        # last name followed by first name
        variations.append(f"{last_name} {first_name}")
        
        # first name with last name only
        variations.append(f"{first_name} {last_name}")
    
    # make sure names are unique and return
    return list(set(variations))


def match_name(name_a, name_b, normalize=False, scorer=rapidfuzz.distance.Indel.normalized_similarity, threshold=0):
    """
    Compare two names and evaluate their similarity based on generated variations of the longer name.
    
    Parameters:
    -----------
    name_a: str
        First name to compare.
    name_b: str
        Second name to compare.
    normalize: bool, (optional)
        If True, normalizes the names to remove diacritic marks (accents). Default is False.
    scorer: callable, (optional)
        Scoring function from RapidFuzz library to calculate name similarity. Defaults is Indel.normalized_similarity.
    threshold: int, (optional)
        Minimum score to consider as a match from the scorer function. Defaults is 0.
    
    Returns:
    --------
    int
        The highest similarity score found if matches are above the threshold. Otherwise 0.
    
    Raises:
    -------
    TypeError
        If either `name_a` or `name_b` is not a string.
    """
    if not isinstance(name_a, str) or not isinstance(name_b, str):
        raise TypeError("Both names must be strings.")
    
    # normalize the names to use unicode characters
    if normalize:
        category = unicodedata.category
        name_a = ''.join([c for c in unicodedata.normalize('NFD', name_a) if category(c) != 'Mn'])
        name_b = ''.join([c for c in unicodedata.normalize('NFD', name_b) if category(c) != 'Mn'])
    
    # set the longer name to be used to generate variations
    vary_name = name_a if len(name_a) > len(name_b) else name_b
    compare_name = name_a if vary_name != name_a else name_b
    
    # create name variations for comparison
    name_variations = generate_name_variations(vary_name)
    
    # evaluate the match
    fuzz_output = rapidfuzz.process.extractOne(compare_name, name_variations, score_cutoff=threshold, scorer=scorer)
    if fuzz_output is None:
        return 0
    else:
        _, score, _ = fuzz_output
        return score