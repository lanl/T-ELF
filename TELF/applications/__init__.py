import sys
sys.path += ["Cheetah"]
sys.path += ["Bunny"]
sys.path += ["Penguin"]

# Cheetah
from .Cheetah.cheetah import Cheetah

# Bunny
from .Bunny.bunny import Bunny, BunnyFilter, BunnyOperation
from .Bunny.auto_bunny import AutoBunny, AutoBunnyStep

# Penguin
from .Penguin.penguin import Penguin