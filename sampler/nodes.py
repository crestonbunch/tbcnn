"""Parse nodes from a given data source."""

import ast
import cPickle as pickle
from collections import defaultdict

def parse(args):
    """Parse nodes with the given args."""
    print ('Loading pickle file')
    
    with open(args.infile, 'rb') as file_handler:
        data_source = pickle.load(file_handler)
    print ('Pickle load finished')

    node_counts = defaultdict(int)
    samples = []

    has_capacity = lambda x: args.per_node < 0 or node_counts[x] < args.per_node
    can_add_more = lambda: args.limit < 0 or len(samples) < args.limit

    for item in data_source:
        root = item['tree']
        new_samples = [
            {
                'node': _name(root),
                'parent': None,
                'children': [_name(x) for x in ast.iter_child_nodes(root)]
            }
        ]
        gen_samples = lambda x: new_samples.extend(_create_samples(x))
        _traverse_tree(root, gen_samples)
        for sample in new_samples:
            if has_capacity(sample['node']):
                samples.append(sample)
                node_counts[sample['node']] += 1
            if not can_add_more:
                break
        if not can_add_more:
            break
    print ('dumping sample')

    with open(args.outfile, 'wb') as file_handler:
        pickle.dump(samples, file_handler)
        file_handler.close()

    print('Sampled node counts:')
    print(node_counts)
    print('Total: %d' % sum(node_counts.values()))

def _create_samples(node):
    """Convert a node's children into a sample points."""
    samples = []
    for child in ast.iter_child_nodes(node):
        sample = {
            "node": _name(child),
            "parent": _name(node),
            "children": [_name(x) for x in ast.iter_child_nodes(child)]
        }
        samples.append(sample)

    return samples

def _traverse_tree(tree, callback):
    """Traverse a tree and execute the callback on every node."""

    queue = [tree]
    while queue:
        current_node = queue.pop(0)
        children = list(ast.iter_child_nodes(current_node))
        queue.extend(children)
        callback(current_node)

def _name(node):
    """Get the name of a node."""
    return type(node).__name__
