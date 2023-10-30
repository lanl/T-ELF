import sys
sys.path += ["decompositions/utilities/"]
sys.path += ["utilities/"]
sys.path += ["decompositions/"]

from .NMFk import NMFk
from .RESCALk import RESCALk
from .TriNMFk import TriNMFk

from .decompositions.nmf_recommender import RNMFk_predict
