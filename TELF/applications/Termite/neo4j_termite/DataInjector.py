from .constants import *
import numpy as np, pandas as pd, ast
from neo4j import GraphDatabase 
from copy import deepcopy
from tqdm import tqdm
import math
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

class InjectorNeo4j:
    def __init__(self,
                 kg_credentials=None,
                 verbose=False):
        """
        Termite, Knowledge graph builder tool.

        Parameters
        ----------
        verbose : bool, optional
            Verbosity flag. The default is False.
        kg_credentials : tuple[str, tuple[str,str]]
            first string is url of graph
            second tuple is auth, user and password
        
        Returns
        -------
        None.
        """
        self.kg_credentials = kg_credentials
        self.verbose = verbose

    def setGraphCredentials(self,credentials):
        """
        Set the knowledge graph credentials.
        
        Parameters:
        -----------
        credentials : tuple[str, tuple[str,str]]
            first string is url of graph
            second tuple is auth, user and password

        Returns:
        --------
        None
        """
        self.kg_credentials = credentials

    def getGraphCredentials(self):
        """
        set the knowledge graph credentials
        
        Parameters:
        -----------
        None

        Returns:
        --------
        tuple[str, tuple[str,str]]
            first string is url of graph
            second tuple is auth, user and password
        """
        return self.kg_credentials


    def add_triple(self, 
                   data_container,  # Dictionary containing data
                   head=np.nan,  # Head node value
                   head_type=np.nan,  # Head node type
                   head_attributes=np.nan,  # Head node attributes
                   tail=np.nan,  # Tail node value
                   tail_type=np.nan,  # Tail node type
                   tail_attributes=np.nan,  # Tail node attributes
                   relation=np.nan,  # Relation between head and tail
                   weight=np.nan):  # Weight of the relation
        """
        Add a triple to the data_container dictionary.

        Parameters:
        -----------
        data_container : dict
            Dictionary containing data.
        head : object, optional
            Head node value. The default is np.nan.
        head_type : object, optional
            Head node type. The default is np.nan.
        head_attributes : object, optional
            Head node attributes. The default is np.nan.
        tail : object, optional
            Tail node value. The default is np.nan.
        tail_type : object, optional
            Tail node type. The default is np.nan.
        tail_attributes : object, optional
            Tail node attributes. The default is np.nan.
        relation : object, optional
            Relation between head and tail. The default is np.nan.
        weight : object, optional
            Weight of the relation. The default is np.nan.
        """
        # Append data to data_container
        data_container[H].append(head)
        data_container[HT].append(head_type)
        data_container[HA].append(head_attributes)
        data_container[TT].append(tail_type)
        data_container[T].append(tail)
        data_container[R].append(relation)
        data_container[W].append(weight)
        data_container[TA].append(tail_attributes)

    def make_unique_constrains(self,column_triplet_map= None, verbose=False, additional_uniques=None):
        if not column_triplet_map:
            raise ValueError("Need a map for contraints")
        
        entities = column_triplet_map.get("ENTITIES")
        if not entities:
            raise ValueError("Need entities in the map")

        if verbose:
            print(entities)
        
        URI, AUTH = self.getGraphCredentials()
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            for entity in entities:
                make_unique = entity.get(MAKE_ID_UNIQUE)
                if make_unique:
                    Head_type = entity.get(ET)
                    unique_contraint = f"CREATE CONSTRAINT {Head_type}_id_unique FOR (n:{Head_type}) REQUIRE n.id IS UNIQUE"
                    try:

                        driver.execute_query(unique_contraint,
                            database_="neo4j",
                        )
                    except Exception as e:
                        print(f"Failed to create unique constraint on: \n\t{unique_contraint} \n\t {e}")

    def from_csv_to_triplets(self, 
                             csv_path, 
                             save_path,  
                             column_triplet_map=None):
        """
        Builds a datafile that maps the raw csv into a head-relation-tail csv
        
        Parameters:
        -----------
        csv_path : str
            path to raw data
        save_path : str
            path to save mapped data
        column_triplet_map : dict
            Has entities and relations as keys and has a form that is of
            the following where the subkeys are defined in the constants file   :
                {'ENTITIES':[{  ET:TYPE, FROM_COL: COL },],
                'RELATIONS':[{  HT:TYPE, R:TYPE, TT:TYPE, 
                                EXTRACT_H:get_H, EXTRACT_T: get_T },]}

        Returns:
        --------
        None
        """
        verbose = self.verbose
        df = pd.read_csv(csv_path)

        docs = {H:[], R:[], T:[], W:[], HT:[], TT:[], HA:[], TA:[]}

        for index, data in tqdm(df.iterrows(), total = len(df)):
            row_entities = {}
            # EXTRACT ENTITIES
            for entity_map in column_triplet_map['ENTITIES']:

                # Added to unique contraints can be added in the map without prcessing as a column operator 
                # These entities are expected to have a function to extract them in the relations
                if FROM_COL not in entity_map:
                    continue
                    
                extraction_function = entity_map.get(EXTRACT_ENTITY)
                if verbose:
                    print(entity_map)

                # If there is an extraction function, assume the function will also handle entity attributes
                if extraction_function:
                    args = entity_map.get(ARGS)
                    if args:
                        if 'col' in args:  # get the column value, remove col name from data
                            args['data'] = data[args['col']]
                            del args['col']
                        entity = extraction_function(args)
                    else:
                        entity = extraction_function()
                

                else:
                    entity = deepcopy(RETURN_TYPE)
                    if verbose:
                        print(f"entity_map[FROM_COL] ={entity_map[FROM_COL]}" )

                    if entity_map[FROM_COL] == ROW_INDEX:
                        entity[ENTITY] = int(index)
                        if verbose:
                            print('entity_map[FROM_COL] == ROW_INDEX')
                    else:
                        entity[ENTITY] = data[entity_map[FROM_COL]]

                    attribute_cols = entity_map.get(ATTR_COL)
                    if attribute_cols:
                        for attribute in attribute_cols:
                            attribute_function = attribute.get(RETREIVAL)
                            if attribute_function:
                                # TODO: implement
                                pass
                            else:
                                attr_value = data[attribute[FROM_COL]]

                                if type(attr_value) == str or type(attr_value) == int or (type(attr_value) == float and not math.isnan(attr_value)):
                                    if  entity[ATTRIBUTES] == None:
                                        entity[ATTRIBUTES] = [(attribute[ATTR_NAME], attr_value)]
                                    else:
                                        entity[ATTRIBUTES].append((attribute[ATTR_NAME], attr_value))
                                # else:
                                #     print("Skipped attribute")

                    if verbose:
                        print(f"entity={entity} for entity_map[ET] ={entity_map[ET]}")


                row_entities[entity_map[ET]] = entity 
            
            if verbose:
                print(f"row_entities={row_entities}")
                
            # EXTRACT RELATIONS
            for relation_map in column_triplet_map['RELATIONS']:
                triple_details = {H:None,
                                    T:None,
                                    HT: relation_map[HT],
                                    TT: relation_map.get(TT),
                                    HA:None,
                                    TA:None,
                                    W:None,               
                                    R: relation_map.get(R),
                                    }
                
                head_extraction = relation_map.get(EXTRACT_H)
                head_args = relation_map.get(ARGS_H, {})
                head_args['data'] = data

                tail_extraction = relation_map.get(EXTRACT_T)
                tail_args = relation_map.get(ARGS_T, {})
                tail_args['data'] = data

                # The heads and tails are embedded inline of the data row, and both must be exctracted through the functions passed
                if head_extraction and tail_extraction:
                    head_entities = head_extraction(head_args)  # extraction call, pass head type, must return a list of RETURN_TYPE
                    tail_entities = tail_extraction(tail_args)  # extraction call, pass tail type, must return a list of RETURN_TYPE

                    pairing = relation_map.get(PAIRING)
                    if pairing == MANY_TO_MANY or not pairing: # default if no pairing specified
                        for i, head_entity in enumerate(head_entities):
                            triple_details[H] = head_entity[ENTITY]    
                            triple_details[W] = head_entity[W]    
                            triple_details[HA] = row_entities[triple_details[HT]][ATTRIBUTES]    

                            for tail_entity in tail_entities:
                                if  i > 0:
                                    triple_details[HA] = None
                                triple_details[T] = tail_entity[ENTITY]    
                                triple_details[TA] = tail_entity[ATTRIBUTES]   
                                triple_details[W] = tail_entity[W]  

                                self.add_triple(docs, **triple_details)

                    else: # index pairing
                        heads_len = len(head_entities)
                        tails_len = len(tail_entities)
                        if verbose:
                            print(head_entities)
                            print(tail_entities)
                            print(f"heads_len ={heads_len}, tails_len={tails_len}")
                        assert heads_len == tails_len
                        for head_entity, tail_entity in zip(head_entities, tail_entities):
                            triple_details[H] = head_entity[ENTITY]    
                            triple_details[HA] = head_entity[ATTRIBUTES]    
                            triple_details[W] = head_entity[W]    

                            triple_details[T] = tail_entity[ENTITY]    
                            triple_details[TA] = tail_entity[ATTRIBUTES]   
                            triple_details[W] = tail_entity[W]  

                        if triple_details[H] not in [None, 'None'] and triple_details[T] not in [None, 'None']:
                            self.add_triple(docs, **triple_details)

                # The heads only are embedded inline of the data row, and must be exctracted through the function passed
                elif head_extraction:
                    head_entities = head_extraction(head_args)  # extraction call,  must return a list of RETURN_TYPE

                    if triple_details[TT]:
                        triple_details[T] = row_entities[triple_details[TT]][ENTITY]   
                        triple_details[TA] = row_entities[triple_details[TT]][ATTRIBUTES]  

                    
                    for i, head_entity in enumerate(head_entities):
                        if i > 0 and triple_details[TT]:
                            triple_details[TA] = None

                        triple_details[H] = head_entity[ENTITY]   
                        triple_details[HA] = head_entity[ATTRIBUTES]    
                        triple_details[W] = head_entity[W]    
                    
                        if triple_details[H] not in [None, 'None'] and triple_details[T] not in [None, 'None']:
                            self.add_triple(docs, **triple_details)

                # The tails only are embedded inline of the data row, and must be exctracted through the function passed
                elif tail_extraction:
                    tail_entities = tail_extraction(tail_args )  # extraction call, must return a list of RETURN_TYPE
                    triple_details[H] = row_entities[triple_details[HT]][ENTITY]   # Head is the Row entity looked up through the head type in the details
                    triple_details[HA] = row_entities[triple_details[HT]][ATTRIBUTES]    


                    # print(f'extracting tail for {triple_details[H] }')

                    for i , tail_entity in enumerate(tail_entities):
                        if i > 0:
                            triple_details[HA] = None
                    

                        triple_details[T] = tail_entity[ENTITY]
                        triple_details[W] = tail_entity[W]
                        triple_details[TA] = tail_entity[ATTRIBUTES]

                        if triple_details[H] not in [None, 'None'] and triple_details[T] not in [None, 'None']:
                            # print('Adding triplet')
                            self.add_triple(docs, **triple_details)
                
                # The head and tail are both accessed directly through their column maps
                else:
                    if verbose:
                        print(f"triple_details[HT] = {triple_details[HT]}")
                    triple_details[H] = row_entities[triple_details[HT]][ENTITY]   # Head is the Row entity looked up through the head type in the details
                    triple_details[HA] = row_entities[triple_details[HT]][ATTRIBUTES]    
                    if triple_details[TT]:
                        triple_details[T] = row_entities[triple_details[TT]][ENTITY]   
                        triple_details[TA] = row_entities[triple_details[TT]][ATTRIBUTES]  
                        triple_details[W] = row_entities[triple_details[TT]][W]    
                    if verbose:
                        print(f"triple_details = {triple_details}")

                    # if triple_details[H] not in [None, 'None'] and triple_details[T] not in [None, 'None']:
                    self.add_triple(docs, **triple_details)

        KG_df = pd.DataFrame.from_dict(docs)
        KG_df.to_csv(save_path, index=False)


    def make_attribute_string(self, 
                              attributes, 
                              node_id):
        """
        Attribute query contructor for nodes to set the attribuets in the KG
        
        Parameters:
        -----------
        attributes : list
            list of tuples containing the attribute name/id and its value
        node_id:
            which node to assign the attribute

        Returns:
        --------
        str
            part of a query containing attributes to be appended to a larger query
        """
        query_part = ''
        attributes =  ast.literal_eval(attributes)
        if len(attributes):
            for attribute in attributes:
                attribute_identifier  = attribute[0]
                attribute_value =  attribute[1]
                query_part +=  "SET "+node_id+"." + attribute_identifier + "='" + str(attribute_value).replace('"', '\\"').replace('\'', '\\\'').replace('\\\\\'', '\\\'') + "'"
        return query_part


    def make_triples(self, 
                     driver, 
                     node1, 
                     node1_type, 
                     relation, 
                     node2, 
                     node2_type, 
                     weight, 
                     head_attributes, 
                     tail_attributes, 
                     index = 0):
        """
        Parses data into query format for Knowledge graph. Injects query in final step using the driver.
        
        Parameters:
        -----------
        driver : KG driver
            to operate the queries
        node1 : str
            The ID of node 1
        node1_type : str
            The graph label for node 1
        relation : str
            The edge's graph label
        node2 : str
            The ID of node 
        node2_type : str
            The graph label for node 2
        weight : float
            Weight for the edge
        head_attributes : list
            Attributes to assign to the head node
        tail_attributes : list
            Attributes to assign to the tail node
        index : int
            Current index of Head-Entity-Tail iteration. Helps identify issues in data

        Returns:
        --------
        None, int
            returns the negative index of queries with problems. No issue, nothing is returned.
        """
        # MAKE SURE HEAD NOT NULL
        for val in [node1, node1_type]:
            if type(val) == float and math.isnan(val):
                return -index
        
        # MAKE ID INT IF NOT STRING
        try:
            node1 = int(node1)
        except:
            pass

        # GENERATE HEAD AND ITS ATTRIBUTES
        query = "MERGE (a:"+node1_type+" {id: $node1} )"
        if type(head_attributes) == str: # not math.isnan(head_attributes):
            query += self.make_attribute_string(head_attributes, node_id='a')

        # IF TAIL AND RELATION ARE NOT NULL< GENERATE THEM
        skip_relation_tail = False
        for val in [relation, node2, node2_type]:
            if type(val) == float and math.isnan(val):
                skip_relation_tail = True
        
        if not skip_relation_tail:
            try:
                node2 = int(node2)
            except:
                pass
            
            # GENERATE TAIL AND ITS ATTRIBUTES
            query += "MERGE (b:"+node2_type+" {id: $node2} )"
            if type(tail_attributes) == str: #not math.isnan(tail_attributes):
                query += self.make_attribute_string(tail_attributes, node_id='b')

            # RELATION and optional WEIGHT
            if math.isnan(weight):
                query +=  "MERGE (a)-[:"+relation+" ]->(b)"
            else:
                query +=  "MERGE (a)-[:"+relation+" {weight: $weight} ]->(b)"


        # Makes EDGES and NODES in NEO4J
        driver.execute_query(query,
            node1=node1,  node2=node2,  weight=weight,
            database_="neo4j",
        )


    def iterate_csv_triplets_into_graph(self, 
                                        triplets_path, 
                                        start_from =0, 
                                        args={}):
        """
        Iterates parsed data to call the injection to graph function.

        Parameters:
        -----------
        triplets_path : str
            Path to the mapped data
        args : dict
            any additions to be made

        Returns:
        --------
        list
            Indicies of failed injections
        """

        df = pd.read_csv(triplets_path)

        if start_from:
            start = int(len(df)* start_from)
            df = df.iloc[start:]
            
        failed_indicies = []

        URI, AUTH = self.getGraphCredentials()
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            for index, data in tqdm(df.iterrows(), total = len(df)):
            
                node1 = data[H]
                node2 = data[T]
                
                node1_type = data[HT]
                node2_type = data[TT]

                relation = data[R]
                weight = data[W]

                head_attributes = data.get(HA, 'nan')
                tail_attributes = data.get(TA, 'nan')
                index_on_fail = self.make_triples(driver, node1, node1_type, relation, node2, node2_type, weight,head_attributes, tail_attributes, index)
                if index_on_fail:
                    failed_indicies.append(index_on_fail)

        return failed_indicies
 
    def process_row(self, 
                    data, 
                    index, 
                    driver):
        """
        Process a row of data and make triples using the provided driver.
        
        :param data: The data dictionary containing information about nodes, relations, and attributes.
        :param index: The index of the row being processed.
        :param driver: The driver object for interacting with the database.
        :return: The triples generated by the function call to make_triples.
        """
        
        node1 = data[H]
        node2 = data[T]

        node1_type = data[HT]
        node2_type = data[TT]

        relation = data[R]
        weight = data[W]

        head_attributes = data.get(HA, 'nan')
        tail_attributes = data.get(TA, 'nan')

        return self.make_triples(driver, node1, node1_type, relation, node2, node2_type, weight, head_attributes, tail_attributes, index)

    def update_database_multithreaded(self,
                                      triplets_path,
                                      start_from=0,
                                      shuffle_rows=True):
        """
        Iterates parsed data to call the injection to graph function.

        The code does not handle iterations with shuffle_rows=True and start_from != 0.
        If shuffle_rows with start_from is needed, manually modify data to accomodate then repass the moddified path.

        Parameters:
        -----------
        triplets_path : str
            Path to the mapped data
        start_from : int
            Which row to start from
        shuffle_rows : bool
            Random shuffle of rows to prevent Neo4j from deadlocking on rows that have common nodes

        Returns:
        --------
        list
            Indicies of failed injections
        """
        df = pd.read_csv(triplets_path)
        
        if shuffle_rows:
            df = df.sample(frac=1).reset_index(drop=True)
            df.to_csv(triplets_path)
        
        failed_indices = []

        URI, AUTH = self.getGraphCredentials()
        
        if start_from:
            start = int(len(df)* start_from)
            print(f"skipping {start}")
            df = df.iloc[start:]
            
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_row, data, index, driver) for index, data in df.iterrows()]
                # print(len(futures))
                for future in tqdm(as_completed(futures), total=len(futures)):
                
                    index_on_fail = future.result()
                    if index_on_fail:
                        failed_indices.append(index_on_fail)

        return failed_indices


    def write_csv(self, filename, data, headers):
        """
        Writes data to a CSV file with specified headers.

        Parameters:
        -----------
        filename : str
            Path to the CSV file.
        data : list of dict
            Data to be written to the CSV file.
        headers : list of str
            Headers for the CSV file.

        Returns:
        --------
        None
        """

        # Open the CSV file in 'w' mode with UTF-8 encoding
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Create a DictWriter object from the opened file
            writer = csv.DictWriter(csvfile, fieldnames=headers)

            # Write the headers to the CSV file
            writer.writeheader()

            # Iterate over the data and write each row to the CSV file
            for row in data:
                writer.writerow(row)

    def process_json_to_csv(self, 
                            json_path):
        """
        Process a JSON file containing nodes and relationships and write 
        them to CSV files.

        Parameters:
        -----------
        json_path : str
            Path to the JSON file.

        Returns:
        --------
        None
        """

        # Lists to store node and relationship data
        nodes_data = []
        relationships_data = []

        # Open and iterate through the file line by line
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # Parse the line as JSON
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip lines that can't be parsed as JSON
                
                # If the object is a node, extract its data
                if obj['type'] == 'node':
                    properties = obj.get('properties', {})
                    properties[':ID'] = obj.get('id')
                    properties[':LABEL'] = ';'.join(obj.get('labels', []))
                    nodes_data.append(properties)
                # If the object is a relationship, extract its data
                elif obj['type'] == 'relationship':
                    properties = obj.get('properties', {})
                    properties[':START_ID'] = obj.get('start', {}).get('id')
                    properties[':END_ID'] = obj.get('end', {}).get('id')
                    properties[':TYPE'] = obj.get('label')
                    relationships_data.append(properties)

        # Define CSV headers dynamically based on collected keys
        node_headers = list(set(key for row in nodes_data for key in row))
        relationship_headers = list(set(key for row in relationships_data for key in row))

        # Write node data to a CSV file
        self.write_csv('nodes.csv', nodes_data, node_headers)
        # Write relationship data to a CSV file
        self.write_csv('relationships.csv', relationships_data, relationship_headers)
