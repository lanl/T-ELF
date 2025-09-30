class ReturnEntities:
    def __init__(self):
        self.all_returns = []

    def add_ent(self, entity, attributes=None):
        """
        Adds an entity to the list of returns.

        Parameters:
        -----------
        entity : any
            The entity to be added.
        attributes : list of tuples, optional
            The attributes of the entity. The default is None.

        Returns:
        --------
        None

        """
        # Create a dictionary representing the entity and its attributes
        entity_dict = {
            "ENTITY": entity,
            "ATTRIBUTES": attributes if attributes is not None else []
        }
        # Append the entity dictionary to the list of returns
        return self.all_returns.append(entity_dict)


    def returns(self):
        """
        Returns the list of entities.

        Returns:
        --------
        list
            A list of dictionaries, where each dictionary contains an entity and its attributes.
        """
        # Return the list of entities
        return self.all_returns
