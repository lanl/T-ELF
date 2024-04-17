def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def compute_levenshtein_distances_manual(input_dict):
    keys = list(input_dict.keys())
    distances = {}
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1 = keys[i]
            key2 = keys[j]
            distance = levenshtein_distance(input_dict[key1], input_dict[key2])
            distances[(key1, key2)] = distance
            
    return distances

def flatten_acronym_dict(input_dict):
    flat_dict = {}