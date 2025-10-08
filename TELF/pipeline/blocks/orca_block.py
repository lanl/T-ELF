from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional, Sequence
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...pre_processing import Orca
from ...pre_processing.Orca.utils import orca_summary
from ...helpers.frames import add_num_known_col, prep_affiliations, drop_columns_if_exist
from ...helpers.filters import clean_affiliations

class OrcaBlock(AnimalBlock):

    CANONICAL_NEEDS = ('df', )

    def __init__(
        self,
        needs = CANONICAL_NEEDS,
        provides = ('df', 'map'),
        tag = 'Orca',
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ) -> None:
        
        default_init = {'verbose':True}
        default_call = {}

        super().__init__(
            needs = needs,
            provides = provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            conditional_needs=conditional_needs,
            tag=tag,
            **kw,
        )


    def run(self, bundle: DataBundle) -> None:
        raw = bundle[self.needs[0]]
        if isinstance(raw, (str, Path)):
            df = self.load_path(raw)
        else:
            df = raw.copy()
            
        df = drop_columns_if_exist(df, cols = ['slic_affiliations', 'slic_author_ids', 'slic_authors'])

        orca = Orca(**self.init_settings)

        
        for col in ["affiliations", "year"]:
            if col not in df.columns:
                df[col] = pd.NA   
                
        df = clean_affiliations(df)
        orca_map_df = orca.run(df)
        df = clean_affiliations(df)
        if df.get('affiliations', pd.Series(dtype=object)).notna().any():
            df = clean_affiliations(df)
        orca_map_df = orca.run(df)

        # orca_map_df = orca_map_df.dropna(subset=['scopus_ids']).reset_index(drop=True)
        if orca_map_df.get('scopus_ids', pd.Series(dtype=object)).notna().any():
            orca_map_df = orca_map_df.dropna(subset=['scopus_ids']).reset_index(drop=True)
        elif orca_map_df.get('s2_ids', pd.Series(dtype=object)).notna().any():
            orca_map_df = orca_map_df.dropna(subset=['s2_ids']).reset_index(drop=True)
        orca_map_df = add_num_known_col(orca_map_df)

        orca_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        orca_dir.mkdir(parents=True, exist_ok=True)


        df = orca.apply(df)
        # df = clean_affiliations(df)
        # df = prep_affiliations(df)
        if df.get('slic_affiliations', pd.Series(dtype=object)).notna().any():
            df = clean_affiliations(df)
            df = prep_affiliations(df)
        

        if 'type' in df.columns:
            orca_summary_path = save_path=orca_dir / 'type_summary.csv'
            summary_df = orca_summary(df, save_path=orca_summary_path)

        orca_map_df_path = orca_dir / 'orca_map.csv'
        orca_map_df.to_csv(orca_map_df_path, index=False)

        orca_df_path = orca_dir / 'orca_df.csv'
        df.to_csv(orca_df_path, index=False)
        
        self.register_checkpoint(self.provides[0], orca_df_path)
        self.register_checkpoint(self.provides[1], orca_map_df_path)

        bundle[f"{self.tag}.{self.provides[0]}"] = df
        bundle[f"{self.tag}.{self.provides[1]}"] = orca_map_df
