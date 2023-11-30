class VultureModuleBase:
    
    def __init__(self, frozen=None, max_display=25):
        # store the initialization parameters
        self.frozen = frozen
        self.max_display = max_display
        
    def __repr__(self):
        classname = self.__class__.__name__
        attributes = ', '.join(f"{key}={self._format_attr(value)}" for key, value in self.__dict__.items()
                              if key not in {'max_display'} and not key.startswith('_'))
        return f"{classname}({attributes})"

    def _format_attr(self, value):
        if isinstance(value, list):
            return self._format_list(value)
        return repr(value)

    def _format_list(self, lst):
        if len(lst) > self.max_display:
            return f"[{', '.join(repr(x) for x in lst[:self.max_display])}, ... (+{len(lst) - self.max_display} more)]"
        return repr(lst)
    
    def _should_preserve(self, token):
        """Check if the token should be preserved and not cleaned."""
        return token in self.frozen
    
    def run(self, text):
        raise NotImplementedError("Subclasses must implement this method!")

    
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
