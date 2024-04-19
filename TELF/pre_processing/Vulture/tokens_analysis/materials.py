from itertools import permutations 
import re

def expand_materials_regex(material:str):
    """
    Expand a given material string by finding all the elements that match the regex pattern and generating all possible permutations of those elements.

    Parameters
    ----------
    material : str
        The material string to be expanded.

    Returns
    -------
    list
        A list of all possible permutations of the elements in the material string, with duplicates removed.
    """
    element_pattern = r'[A-Z][a-z]?[0-9]*'
    elements = re.findall(element_pattern, material)
    
    permuted_elements = permutations(elements)
    permuted_materials = [''.join(perm) for perm in permuted_elements]
    
    return list(set(permuted_materials))  # Remove duplicates and return

def permute_material_list(materials:list,  save_path:str ,sort=True,):
    """
    Generates a list of permuted materials based on the given list of materials.

    Parameters
    ----------
    materials : list
        A list of materials to permute.
    save_path : str
        The path to save the permuted materials. If None, the permuted materials will be returned.
    sort : bool, optional
        Whether to sort the materials before permuting. Defaults to True.

    Returns
    -------
    list
        A list of permuted materials.
    """
    if sort:
        materials.sort()
    
    permuted_materials = []
    for material in materials:
        permuted_materials += expand_materials_regex(material)

    if save_path:
        with open(save_path, 'w') as f:
            for permuted_material in permuted_materials:
                f.write(permuted_material + '\n')
    else:
        return permuted_materials
