class VultureModuleBase:
    
    def __init__(self, frozen=None):
        # store the initialization parameters
        self.init_args = (frozen,)
        self.init_kwargs = {}
        self.frozen = frozen

    def _should_preserve(self, token):
        """Check if the token should be preserved and not cleaned."""
        return token in self.frozen
    
    def run(self, text):
        raise NotImplementedError("Subclasses must implement this method!")
        
    def get_init_params(self):
        """Retrieve initialization parameters."""
        return {
            'args': self.init_args, 
            'kwargs': self.init_kwargs
        }

    
    # GETTERS / SETTERS

    
    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def frozen(self, frozen):
        if frozen is None:
            self._frozen = set()
        elif isinstance(frozen, set):
            self._frozen = frozen
        elif isinstance(frozen, list):
            self._frozen = set(frozen)
        else:
            raise TypeError(f'Unexpected type "{type(frozen)}" for `frozen`! Expected list, set, or None!')
