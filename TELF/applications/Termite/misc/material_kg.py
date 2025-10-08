# combined_material_clusters_path =  "./data/material_injection.csv"
# combined_material_clusters_path =  "./affiliation_resolve_materials.csv"
from TELF.applications import Termite
from TELF.applications.Termite.neo4j_termite import *
termite = Termite()
aflow_path =  "./material_properties.csv"
# mats_and_NER_path = "./materials_and_MATNER_triplets.csv"
aflow_trip_path = "./aflow_triplets.csv"

ICSD_TYPE = 'ICSD'
CRYSTAL_SYSTEM_TYPE = 'Crystal_system'
CRYSTAL_CLASS_TYPE ='Crystal_class'
PEARSON_TYPE = 'Pearson'
SPACEGROUP_TYPE = 'Spacegroup'

MATERIAL_ICSD_RELATION = 'crystal_database_id' 
ICSD_SYSTEM_RELATION = 'id_is_system'
ICSD_CLASS_RELATION ='id_is_class'
ICSD_PEARSON_RELATION ='id_is_pearson'
ICSD_SPACEGROUP_RELATION ='id_is_spacegroup'

material_triplet_map =  {
    'ENTITIES':
    [
        {ET:MATERIAL_TYPE, FROM_COL: 'material', MAKE_ID_UNIQUE:True},     
        {ET:ICSD_TYPE, FROM_COL: 'ICSD',  MAKE_ID_UNIQUE:True,ATTR_COL:[{FROM_COL: 'geometry', ATTR_NAME:'geometry'}, {FROM_COL: 'Egap', ATTR_NAME:'Egap'} ]},                     
        {ET:CRYSTAL_SYSTEM_TYPE, FROM_COL: 'crystal_system',  MAKE_ID_UNIQUE:True},                     
        {ET:CRYSTAL_CLASS_TYPE, FROM_COL: 'crystal_class',  MAKE_ID_UNIQUE:True},                     
        {ET:PEARSON_TYPE, FROM_COL: 'pearson',  MAKE_ID_UNIQUE:True},                     
        {ET:SPACEGROUP_TYPE, FROM_COL: 'spacegroup', MAKE_ID_UNIQUE:True},                     

    ],
    'RELATIONS':
    [   
        {HT:MATERIAL_TYPE, R:MATERIAL_ICSD_RELATION, TT:ICSD_TYPE},
        {HT:ICSD_TYPE, R:ICSD_SYSTEM_RELATION , TT: CRYSTAL_SYSTEM_TYPE,  },
        {HT:ICSD_TYPE, R:ICSD_CLASS_RELATION , TT:CRYSTAL_CLASS_TYPE ,  },
        {HT:ICSD_TYPE, R: ICSD_PEARSON_RELATION, TT: PEARSON_TYPE,  },
        {HT:ICSD_TYPE, R: ICSD_SPACEGROUP_RELATION, TT: SPACEGROUP_TYPE,  },

    ]
}
termite.make_unique_constrains(  material_triplet_map)
termite.from_csv_to_triplets(aflow_path, aflow_trip_path, material_triplet_map)
termite.update_database_multithreaded(aflow_trip_path,start_from=0,shuffle_rows=True )