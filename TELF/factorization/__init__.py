import sys
import os
sys.path += [os.path.join("decompositions", "utilities")]
sys.path += ["utilities"]
sys.path += ["decompositions"]

from .NMFk import NMFk
from .HNMFk import HNMFk
from .RESCALk import RESCALk
from .SymNMFk import SymNMFk
from .TriNMFk import TriNMFk
from .SPLIT import SPLIT
from .SPLITTransfer import SPLITTransfer
from .decompositions.nmf_recommender import RNMFk_predict
