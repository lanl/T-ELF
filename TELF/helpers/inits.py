from ..pre_processing.Vulture.modules import SimpleCleaner
from ..pre_processing.Vulture.modules import LemmatizeCleaner
from ..pre_processing.Vulture.modules import SubstitutionCleaner
from ..pre_processing.Vulture.modules import RemoveNonEnglishCleaner
from .terms import load_stop_terms

def init_vulture_steps(settings):
    """
    Parse the Vulture JSON to create a list of Vulture cleaning steps
    
    Parameters:
    -----------
    settings: dict
        The Vulture settings
        
    Returns:
    --------
    list:
        List of cleaning steps (in order) to be executed by Vulture
    """
    VULTURE_MAP = {
        'SimpleCleaner': SimpleCleaner,
        'LemmatizeCleaner': LemmatizeCleaner,
        'SubstitutionCleaner': SubstitutionCleaner,
        'RemoveNonEnglishCleaner': RemoveNonEnglishCleaner,
    }
    
    steps = []
    for s in settings:
        cleaner = s.get('type')
        if cleaner is None:
            raise ValueError('Config file is missing `type` for Vulture step!')
        if cleaner not in VULTURE_MAP:
            raise ValueError(f'Unknown cleaner "{cleaner}"!')
        
        args = s.get('init_args')
        if args is None:
            raise ValueError('Config file is missing `init_args` for Vulture step!')
        
        # process stop words / stop phrases for SimpleCleaner
        if cleaner == 'SimpleCleaner':
            args['stop_words'] = load_stop_terms(args.get('stop_words'), words=True)
            args['stop_phrases'] = load_stop_terms(args.get('stop_phrases'), words=False)
        
        # create the cleaner object
        steps.append(VULTURE_MAP[cleaner](**args))
    return steps

def organize_required_params(params, required_params={}):
    for key, value in required_params.items():
        params[key] = value

    return params
