# TELF/pipeline/blocks/auto_bunny_block.py

from pathlib import Path
from typing import Any, Dict, Sequence, Optional, Tuple, List, Union

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from TELF.applications.Bunny import AutoBunny, AutoBunnyStep

class AutoBunnySimpleBlock(AnimalBlock):
    CANONICAL_NEEDS = ("df",)

    def __init__(
        self,
        num_hops: int = 5,
        modes: Sequence[str] = ("citations",),
        hop_priority: str = "frequency",
        needs: Tuple[str, str] = CANONICAL_NEEDS,
        provides: Tuple[str, ...] = ("df",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "AutoBunnySimple",
        *,
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        """
        Dataset expansion tool

        always needs      : ('df', 'terms')
        needs 'substitutions' only when use_substitutions=True and use_vulture_steps=False
        needs 'vulture_steps' only when use_vulture_steps=True
        provides          : ('df',)
        tag               : 'AutoBunny' (namespace for its outputs)
        """

        # Defaults for AutoBunny
        default_init: Dict[str, Any] = {
            "s2_key": None,
            "scopus_keys": None,
            "cache_dir": None,
            "verbose": True,
            'use_vulture_cheetah' : False # Core functionality to make it simple -- absolutely no cleaning or filtering
        }

        conds = list(conditional_needs or [])

        # Build initial AutoBunnyStep list
        steps = [
            AutoBunnyStep(
                modes=modes,
                max_papers=5000,
                hop_priority=hop_priority,
                vulture_settings=None,
            )
            for _ in range(num_hops)
        ]

        default_call: Dict[str, Any] = {
            "steps": steps,
            "filter_type": "AFFILCOUNTRY",
            "filter_value": None,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            verbose=verbose,
            **kw,
        )

    # ---------------
    # Block execution
    # ---------------
    def run(self, bundle: DataBundle) -> None:
        df = self.load_path(bundle[self.needs[0]])
        self.output_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ab = AutoBunny(core=df, output_dir=self.output_dir, **self.init_settings)
        ab_df = ab.run(**self.call_settings)

        bundle[f"{self.tag}.{self.provides[0]}"] = ab_df
