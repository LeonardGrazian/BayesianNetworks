

# superclass of all bayesian network exceptions
class NetworkException(Exception):
    pass


# raised when node is expected to be found in network but is missing
class NodeNotFoundException(NetworkException):
    def __init__(self, node):
        self.node = node

    def __str__(self):
        return 'Node {} not found'.format(self.node)


# superclass of all data exceptions
class DataException(Exception):
    pass


# raised when observation doesn't have values for all nodes in network
class IncompleteDataException(DataException):
    def __init__(self, observation):
        self.observation = observation

    def __str__(self):
        return 'Observation does not have values for all nodes: {}'.format(
            self.observation
        ) 
