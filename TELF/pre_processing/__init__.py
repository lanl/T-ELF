import sys
sys.path += ["Vulture"]
sys.path += ["Beaver"]
sys.path += ["Orca"]
sys.path += ["iPenguin"]


# Beaver
from .Beaver.beaver import Beaver
from .Beaver.cooccurrence import co_occurrence
from .Beaver.sppmi import sppmi
from .Beaver.tenmat import unfold
from .Beaver.vectorize import tfidf
from .Beaver.vectorize import count

# Vulture
from .Vulture.vulture import Vulture
from .Vulture.default_stop_words import STOP_WORDS
from .Vulture.default_stop_phrases import STOP_PHRASES

# Orca
from .Orca.orca import Orca

# iPenguin
from .iPenguin.OSTI.osti import OSTI
from .iPenguin.Scopus.scopus import Scopus
from .iPenguin.SemanticScholar.s2 import SemanticScholar