import sys
sys.path += ["Cheetah"]
sys.path += ["Bunny"]
sys.path += ["Penguin"]

# Cheetah
from .Cheetah.cheetah import Cheetah
from .Cheetah.term_formatter import CheetahTermFormatter, convert_txt_to_cheetah_markdown
from .Cheetah.term_generator import SearchTermGenerator

# Bunny
from .Bunny.bunny import Bunny, BunnyFilter, BunnyOperation
from .Bunny.auto_bunny import AutoBunny, AutoBunnyStep

# Penguin
from .Penguin.penguin import Penguin