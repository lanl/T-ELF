import sys
sys.path += ["Wolf"]
sys.path += ["Peacock"]
sys.path += ["SeaLion"]
sys.path += ["Fox"]
sys.path += ["ArcticFox"]

# Wolf
from .Wolf.wolf import Wolf

# Peacock
# No direct class imports

# SeaLion
from .SeaLion.sealion import SeaLion

# Fox
from .Fox.fox import Fox

# ArcticFox
from .ArcticFox.arcticfox import ArcticFox