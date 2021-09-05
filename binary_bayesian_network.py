
from numpy.random import uniform
import networkx as nx


# @param parent_values: list of node's parents' values
# @returns int, index of prob table corresponding to given parent values
def parent_values_to_index(parent_values):
    index = 0
    for parent_num, parent_values in enumerate(parent_values):
        index += parent_values * (2 ** parent_num)
    return index


# @param prob_table_index: int, index in probability table,
#   takes values between 0 and 2^(num_parents) - 1
# @param num_parents: number of node parents
# @returns tuple of parent values corresponding to given prob_table_index,
#   has length equal to # of parents
#   first parent value is stored in first position in tuple
def parent_index_to_values(prob_table_index, num_parents):
    values = []
    for i in range(1, num_parents + 1):
        n = 2 ** i
        n_lower = 2 ** (i - 1)
        values.append((prob_table_index % n) // n_lower)
    return tuple(values)


# @param node_dict: dict of nodes representing network topology
#   keys are nodes, values are a list of the node's parents
# @returns networkx directed graph with given nodes and topology
def node_dict_to_graph(node_dict):
    g = nx.DiGraph()
    g.add_nodes_from(node_dict.keys())
    for node, parents in node_dict.items():
        g.add_edges_from(
            [(parent, node) for parent in parents]
        )
    return g


class BinaryNode:
    # @param parents: a list, each item is a parent node of this node
    # @param prob_table: a list, each item is the probability that
    #   this node is 1 given that its parents have taken some value
    def __init__(self, id, parents, prob_table):
        assert len(prob_table) == 2 ** len(parents)
        self.id = id
        self.parents = parents
        self.prob_table = prob_table

    def __repr__(self):
        return '<BinaryNode {}>'.format(self.id)

    # @param parent_values: list of parents' values
    #   length must match number of node's parents
    #   the first entry of list corresponds to first parent in node.parents
    # @returns a sample of node's value conditioned on parent's values
    def conditional_sample(self, parent_values):
        if uniform() < self.probability(parent_values):
            return 1
        return 0

    # @param parent_values: list of parents' values
    #   length must match number of node's parents
    #   the first entry of list corresp]onds to first parent in node.parents
    # @returns prob that node is 1 given its parents take on parent_values
    def probability(self, parent_values):
        assert len(parent_values) == len(self.parents)
        prob_table_index = parent_values_to_index(parent_values)
        return self.prob_table[prob_table_index]


# TODO: add ability to seed network for deterministic results
class BinaryBayesianNetwork:
    # @param nodes: an iterable containing BinaryNode objects
    def __init__(self, nodes):
        node_dict = {}
        for node in nodes:
            node_dict[node] = node.parents
        self.g = node_dict_to_graph(node_dict)
        # assert that graph is a DAG
        assert len(list(nx.simple_cycles(self.g))) == 0
        self.ordered_nodes = list(nx.topological_sort(self.g))
        self.marginals = {}

    # @param node_values: (optional) a dict of observed node values,
    #   keys are nodes, values are the observed node values
    # @returns a dict of node samples,
    #   keys are nodes, values are the sampled values of those nodes
    #   returns None if any sampled values are not consistent with node_values
    def sample(self, node_values={}):
        node_samples = {}
        for node in self.ordered_nodes:
            parent_values = []
            for p in node.parents:
                parent_values.append(node_samples[p])
            node_sample = node.conditional_sample(parent_values)
            if node in node_values and node_values[node] != node_sample:
                return None
            node_samples[node] = node_sample
        return node_samples

    # @param node_values: a dict of observed node values,
    #   keys are nodes, values are the observed node values
    # @returns a dict of node samples,
    #   keys are nodes, values are the sampled values of those nodes
    def conditional_sample(self, node_values):
        node_samples = self.sample(node_values=node_values)
        while node_samples is None:
            node_samples = self.sample(node_values=node_values)
        return node_samples

    # @param node_values: dict of node values
    #   keys are nodes, values are the value that node takes
    #   if a node is in node_values, all its ancestors must be in node_values
    # @returns the probability of the nodes in node_values
    #   taking on the values in node_values
    def probability(self, node_values):
        node_probs = {}
        for node in self.ordered_nodes:
            if node not in node_values:
                continue
            node_value = node_values[node]
            parent_values = []
            for p in node.parents:
                # TODO: implement custom exception
                # if p not in node_values:
                #     throw custom exception
                parent_values.append(node_values[p])
            node_prob = node.probability(parent_values)
            if node_value == 1:
                node_probs[node] = node_prob
            else:
                node_probs[node] = 1 - node_prob
        p = 1.0
        for node, prob in node_probs.items():
            p *= prob
        return p

    # @param target_nodes: list of nodes in this network
    # @returns list of floats, probability table of target_nodes
    #   indexing works the same as with parent nodes
    #   obtained by brute force computation
    #   O(n_target_nodes * n_ancestors * 2^n_target_nodes * 2^n_ancestors)
    def get_marginal_by_brute_force(self, target_nodes):
        ancestors = set()
        for target_node in target_nodes:
            ancestors.update(nx.algorithms.dag.ancestors(self.g, target_node))
        for target_node in target_nodes:
            if target_node in ancestors:
                ancestors.remove(target_node)
        n_ancestors = len(ancestors)
        n_target_nodes = len(target_nodes)

        prob_sum = [0 for _ in range(2 ** n_target_nodes)]
        for i in range(2 ** n_ancestors):
            ancestor_values = parent_index_to_values(i, n_ancestors)
            node_values = {
                ancestor: ancestor_value
                for ancestor, ancestor_value
                in zip(ancestors, ancestor_values)
            }
            for j in range(2 ** n_target_nodes):
                target_node_values = parent_index_to_values(j, n_target_nodes)
                node_values.update({
                    target_node: target_node_value
                    for target_node, target_node_value
                    in zip(target_nodes, target_node_values)
                })
                prob_sum[j] += self.probability(node_values)
        return prob_sum


def main():
    a = BinaryNode('a', [], [0.5])
    b = BinaryNode('b', [a], [0.9, 0.4])
    bnet = BinaryBayesianNetwork([a, b])
    print(bnet.get_marginal_by_brute_force([a, b]))


if __name__ == '__main__':
    main()
