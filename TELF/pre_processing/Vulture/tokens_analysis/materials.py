import permutations 
import re
def expand_materials_regex(material:str):
    element_pattern = r'[A-Z][a-z]?[0-9]*'
    elements = re.findall(element_pattern, material)
    
    permuted_elements = permutations(elements)
    permuted_materials = [''.join(perm) for perm in permuted_elements]
    
    return list(set(permuted_materials))  # Remove duplicates and return

def permute_material_list(materials:list,  save_path:str ,sort=True,):
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