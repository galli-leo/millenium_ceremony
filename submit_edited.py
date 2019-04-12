def flip_graph(G, n, m, k):
    new = Graph()
    for edge in reversed([e for e in G.edges]):
        v1, v2 = edge
        nv1, nv2 = v1, v2
        if v1[0] != 0 and v1[0] != n+1:
            nv1 = (v1[0], m - v1[1]+1)
        if v2[0] != 0 and v2[0] != n+1:
            nv2 = (v2[0], m - v2[1]+1)
        new.add_edge(v1, v2)
    return new


import collections
import itertools

INFINITY = float('inf')

def hopcroft_karp_matching(G, top_nodes=None):
    def breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                distances[v] = 0
                queue.append(v)
            else:
                distances[v] = INFINITY
        distances[None] = INFINITY
        while queue:
            v = queue.popleft()
            if distances[v] < distances[None]:
                for u in G[v]:
                    if distances[rightmatches[u]] is INFINITY:
                        distances[rightmatches[u]] = distances[v] + 1
                        queue.append(rightmatches[u])
        return distances[None] is not INFINITY

    def depth_first_search(v):
        if v is not None:
            for u in G[v]:
                if distances[rightmatches[u]] == distances[v] + 1:
                    if depth_first_search(rightmatches[u]):
                        rightmatches[u] = v
                        leftmatches[v] = u
                        return True
            distances[v] = INFINITY
            return False
        return True

    left, right = sets(G, top_nodes)
    leftmatches = {v: None for v in left}
    rightmatches = {v: None for v in right}
    distances = {}
    queue = collections.deque()

    num_matched_pairs = 0
    while breadth_first_search():
        for v in left:
            if leftmatches[v] is None:
                if depth_first_search(v):
                    num_matched_pairs += 1

    leftmatches = {k: v for k, v in leftmatches.items() if v is not None}
    rightmatches = {k: v for k, v in rightmatches.items() if v is not None}

    return dict(itertools.chain(leftmatches.items(), rightmatches.items()))



def isolates(G):
    return (n for n, d in G.degree() if d == 0)



def color(G):
    neighbors = G.neighbors

    color = {}
    for n in G:
        if n in color or len(G[n]) == 0:
            continue
        queue = [n]
        color[n] = 1
        while queue:
            v = queue.pop()
            c = 1 - color[v]
            for w in neighbors(v):
                if w in color:
                    if color[w] == color[v]:
                        pass
                else:
                    color[w] = c
                    queue.append(w)

    color.update(dict.fromkeys(isolates(G), 0))
    return color

def sets(G, top_nodes=None):
    if top_nodes is not None:
        X = set(top_nodes)
        Y = set(G) - X
    else:
        c = color(G)
        X = {n for n, is_top in c.items() if is_top}
        Y = {n for n, is_top in c.items() if not is_top}
    return (X, Y)

def arbitrary_element(iterable):
    return next(iter(iterable))

def _simplegraph_eulerian_circuit(G, source):
    degree = G.degree
    edges = G.edges
    vertex_stack = [source]
    last_vertex = None
    while vertex_stack:
        current_vertex = vertex_stack[-1]
        if degree(current_vertex) == 0:
            if last_vertex is not None:
                yield (last_vertex, current_vertex)
            last_vertex = current_vertex
            vertex_stack.pop()
        else:
            _, next_vertex = arbitrary_element(edges(current_vertex))
            vertex_stack.append(next_vertex)
            G.remove_edge(current_vertex, next_vertex)

def eulerian_circuit(G, source=None, keys=False):
    if G.is_directed():
        G = G.reverse()
    else:
        G = G.copy()
    if source is None:
        source = arbitrary_element(G)
    for u, v in _simplegraph_eulerian_circuit(G, source):
        yield u, v



def node_connected_component(G, n):
    return set(_plain_bfs(G, n))

def _plain_bfs(G, source):
    G_adj = G.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G_adj[v])


from collections.abc import Mapping

class AtlasView(Mapping):
    __slots__ = ('_atlas',)

    def __getstate__(self):
        return {'_atlas': self._atlas}

    def __setstate__(self, state):
        self._atlas = state['_atlas']

    def __init__(self, d):
        self._atlas = d

    def __len__(self):
        return len(self._atlas)

    def __iter__(self):
        return iter(self._atlas)

    def __getitem__(self, key):
        return self._atlas[key]

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

    def __str__(self):
        return str(self._atlas)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._atlas)


class AdjacencyView(AtlasView):
    __slots__ = ()

    def __getitem__(self, name):
        return AtlasView(self._atlas[name])

    def copy(self):
        return {n: self[n].copy() for n in self._atlas}

class FilterAtlas(Mapping):
    def __init__(self, d, NODE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK

    def __len__(self):
        return sum(1 for n in self)

    def __iter__(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, key):
        if key in self._atlas and self.NODE_OK(key):
            return self._atlas[key]
        raise KeyError("Key {} not found".format(key))

    def copy(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return {u: self._atlas[u] for u in self.NODE_OK.nodes
                    if u in self._atlas}
        return {u: d for u, d in self._atlas.items()
                if self.NODE_OK(u)}

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self._atlas,
                               self.NODE_OK)

class FilterAdjacency(Mapping):
    def __init__(self, d, NODE_OK, EDGE_OK):
        self._atlas = d
        self.NODE_OK = NODE_OK
        self.EDGE_OK = EDGE_OK

    def __len__(self):
        return sum(1 for n in self)

    def __iter__(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return (n for n in self.NODE_OK.nodes if n in self._atlas)
        return (n for n in self._atlas if self.NODE_OK(n))

    def __getitem__(self, node):
        if node in self._atlas and self.NODE_OK(node):
            def new_node_ok(nbr):
                return self.NODE_OK(nbr) and self.EDGE_OK(node, nbr)
            return FilterAtlas(self._atlas[node], new_node_ok)
        raise KeyError("Key {} not found".format(node))

    def copy(self):
        try:
            node_ok_shorter = 2 * len(self.NODE_OK.nodes) < len(self._atlas)
        except AttributeError:
            node_ok_shorter = False
        if node_ok_shorter:
            return {u: {v: d for v, d in self._atlas[u].items()
                        if self.NODE_OK(v) if self.EDGE_OK(u, v)}
                    for u in self.NODE_OK.nodes if u in self._atlas}
        return {u: {v: d for v, d in nbrs.items() if self.NODE_OK(v)
                    if self.EDGE_OK(u, v)}
                for u, nbrs in self._atlas.items()
                if self.NODE_OK(u)}

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__, self._atlas,
                                   self.NODE_OK, self.EDGE_OK)

class UnionAtlas(Mapping):
    __slots__ = ('_succ', '_pred')

    def __getstate__(self):
        return {'_succ': self._succ, '_pred': self._pred}

    def __setstate__(self, state):
        self._succ = state['_succ']
        self._pred = state['_pred']

    def __init__(self, succ, pred):
        self._succ = succ
        self._pred = pred

    def __len__(self):
        return len(self._succ) + len(self._pred)

    def __iter__(self):
        return iter(set(self._succ.keys()) | set(self._pred.keys()))

    def __getitem__(self, key):
        try:
            return self._succ[key]
        except KeyError:
            return self._pred[key]

    def copy(self):
        result = {nbr: dd.copy() for nbr, dd in self._succ.items()}
        for nbr, dd in self._pred.items():
            if nbr in result:
                result[nbr].update(dd)
            else:
                result[nbr] = dd.copy()
        return result

    def __str__(self):
        return str({nbr: self[nbr] for nbr in self})

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self._succ, self._pred)



class NetworkXException(Exception):
    """Base class for exceptions in NetworkX."""


class NetworkXError(NetworkXException):
    """Exception for a serious error in NetworkX"""



from collections.abc import Mapping, Set

class NodeView(Mapping, Set):
    __slots__ = '_nodes',

    def __getstate__(self):
        return {'_nodes': self._nodes}

    def __setstate__(self, state):
        self._nodes = state['_nodes']

    def __init__(self, graph):
        self._nodes = graph._node

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, n):
        return self._nodes[n]


    def __contains__(self, n):
        return n in self._nodes

    @classmethod
    def _from_iterable(cls, it):
        return set(it)


    def __call__(self, data=False, default=None):
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def data(self, data=True, default=None):
        if data is False:
            return self
        return NodeDataView(self._nodes, data, default)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, tuple(self))

class NodeDataView(Set):
    __slots__ = ('_nodes', '_data', '_default')

    def __getstate__(self):
        return {'_nodes': self._nodes,
                '_data': self._data,
                '_default': self._default}

    def __setstate__(self, state):
        self._nodes = state['_nodes']
        self._data = state['_data']
        self._default = state['_default']

    def __init__(self, nodedict, data=False, default=None):
        self._nodes = nodedict
        self._data = data
        self._default = default

    @classmethod
    def _from_iterable(cls, it):
        try:
            return set(it)
        except TypeError as err:
            if "unhashable" in str(err):
                msg = " : Could be b/c data=True or your values are unhashable"
                raise TypeError(str(err) + msg)
            raise

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        data = self._data
        if data is False:
            return iter(self._nodes)
        if data is True:
            return iter(self._nodes.items())
        return ((n, dd[data] if data in dd else self._default)
                for n, dd in self._nodes.items())

    def __contains__(self, n):
        try:
            node_in = n in self._nodes
        except TypeError:
            n, d = n
            return n in self._nodes and self[n] == d
        if node_in is True:
            return node_in
        try:
            n, d = n
        except (TypeError, ValueError):
            return False
        return n in self._nodes and self[n] == d

    def __getitem__(self, n):
        ddict = self._nodes[n]
        data = self._data
        if data is False or data is True:
            return ddict
        return ddict[data] if data in ddict else self._default

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        if self._data is False:
            return '%s(%r)' % (self.__class__.__name__, tuple(self))
        if self._data is True:
            return '%s(%r)' % (self.__class__.__name__, dict(self))
        return '%s(%r, data=%r)' % \
               (self.__class__.__name__, dict(self), self._data)

class OutEdgeDataView(object):
    __slots__ = ('_viewer', '_nbunch', '_data', '_default',
                 '_adjdict', '_nodes_nbrs', '_report')

    def __getstate__(self):
        return {'viewer': self._viewer,
                'nbunch': self._nbunch,
                'data': self._data,
                'default': self._default}

    def __setstate__(self, state):
        self.__init__(**state)

    def __init__(self, viewer, nbunch=None, data=False, default=None):
        self._viewer = viewer
        self._adjdict = viewer._adjdict
        if nbunch is None:
            self._nodes_nbrs = self._adjdict.items
        else:
            nbunch = list(viewer._graph.nbunch_iter(nbunch))
            self._nodes_nbrs = lambda: [(n, self._adjdict[n]) for n in nbunch]
        self._nbunch = nbunch
        self._data = data
        self._default = default

        if data is True:
            self._report = lambda n, nbr, dd: (n, nbr, dd)
        elif data is False:
            self._report = lambda n, nbr, dd: (n, nbr)
        else:
            self._report = lambda n, nbr, dd: \
                (n, nbr, dd[data]) if data in dd else (n, nbr, default)

    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        return (self._report(n, nbr, dd) for n, nbrs in self._nodes_nbrs()
                for nbr, dd in nbrs.items())

    def __contains__(self, e):
        try:
            u, v = e[:2]
            ddict = self._adjdict[u][v]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, list(self))


class EdgeDataView(OutEdgeDataView):
    __slots__ = ()

    def __len__(self):
        return sum(1 for e in self)

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, dd in nbrs.items():
                if nbr not in seen:
                    yield self._report(n, nbr, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        try:
            u, v = e[:2]
            ddict = self._adjdict[u][v]
        except KeyError:
            try:
                ddict = self._adjdict[v][u]
            except KeyError:
                return False
        return e == self._report(u, v, ddict)

class OutEdgeView(Set, Mapping):
    __slots__ = ('_adjdict', '_graph', '_nodes_nbrs')

    def __getstate__(self):
        return {'_graph': self._graph}

    def __setstate__(self, state):
        self._graph = G = state['_graph']
        self._adjdict = G._succ if hasattr(G, "succ") else G._adj
        self._nodes_nbrs = self._adjdict.items

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    dataview = OutEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._succ if hasattr(G, "succ") else G._adj
        self._nodes_nbrs = self._adjdict.items


    def __len__(self):
        return sum(len(nbrs) for n, nbrs in self._nodes_nbrs())

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (n, nbr)

    def __contains__(self, e):
        try:
            u, v = e
            return v in self._adjdict[u]
        except KeyError:
            return False


    def __getitem__(self, e):
        u, v = e
        return self._adjdict[u][v]


    def __call__(self, nbunch=None, data=False, default=None):
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default)

    def data(self, data=True, default=None, nbunch=None):
        if nbunch is None and data is False:
            return self
        return self.dataview(self, nbunch, data, default)


    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return "{0.__class__.__name__}({1!r})".format(self, list(self))


class EdgeView(OutEdgeView):

    __slots__ = ()

    dataview = EdgeDataView

    def __len__(self):
        num_nbrs = (len(nbrs) + (n in nbrs) for n, nbrs in self._nodes_nbrs())
        return sum(num_nbrs) // 2

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                if nbr not in seen:
                    yield (n, nbr)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        try:
            u, v = e[:2]
            return v in self._adjdict[u] or u in self._adjdict[v]
        except (KeyError, ValueError):
            return False

class DiDegreeView(object):

    def __init__(self, G, nbunch=None, weight=None):
        self._graph = G
        self._succ = G._succ if hasattr(G, "_succ") else G._adj
        self._pred = G._pred if hasattr(G, "_pred") else G._adj
        self._nodes = self._succ if nbunch is None \
            else list(G.nbunch_iter(nbunch))
        self._weight = weight

    def __call__(self, nbunch=None, weight=None):
        if nbunch is None:
            if weight == self._weight:
                return self
            return self.__class__(self._graph, None, weight)
        try:
            if nbunch in self._nodes:
                if weight == self._weight:
                    return self[nbunch]
                return self.__class__(self._graph, None, weight)[nbunch]
        except TypeError:
            pass
        return self.__class__(self._graph, nbunch, weight)

    def __getitem__(self, n):
        weight = self._weight
        succs = self._succ[n]
        preds = self._pred[n]
        if weight is None:
            return len(succs) + len(preds)
        return sum(dd.get(weight, 1) for dd in succs.values()) + \
            sum(dd.get(weight, 1) for dd in preds.values())

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                yield (n, len(succs) + len(preds))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                preds = self._pred[n]
                deg = sum(dd.get(weight, 1) for dd in succs.values()) \
                    + sum(dd.get(weight, 1) for dd in preds.values())
                yield (n, deg)

    def __len__(self):
        return len(self._nodes)

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, dict(self))


class DegreeView(DiDegreeView):

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if weight is None:
            return len(nbrs) + (n in nbrs)
        return sum(dd.get(weight, 1) for dd in nbrs.values()) + \
            (n in nbrs and nbrs[n].get(weight, 1))

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                nbrs = self._succ[n]
                yield (n, len(nbrs) + (n in nbrs))
        else:
            for n in self._nodes:
                nbrs = self._succ[n]
                deg = sum(dd.get(weight, 1) for dd in nbrs.values()) + \
                    (n in nbrs and nbrs[n].get(weight, 1))
                yield (n, deg)



def no_filter(*items):
    return True
class show_nodes(object):
    def __init__(self, nodes):
        self.nodes = set(nodes)

    def __call__(self, node):
        return node in self.nodes
def show_edges(edges):
    alledges = set(edges) | {(v, u) for (u, v) in edges}
    return lambda u, v: (u, v) in alledges



def frozen(*args):
    raise NetworkXError()

def freeze(G):
    G.add_node = frozen
    G.add_nodes_from = frozen
    G.remove_node = frozen
    G.remove_nodes_from = frozen
    G.add_edge = frozen
    G.add_edges_from = frozen
    G.remove_edge = frozen
    G.remove_edges_from = frozen
    G.clear = frozen
    G.frozen = True
    return G

def edge_subgraph(G, edges):
    edges = set(edges)
    nodes = set()
    for e in edges:
        nodes.update(e[:2])
    induced_nodes = show_nodes(nodes)
    induced_edges = show_edges(edges)
    return subgraph_view(G, induced_nodes, induced_edges)
def subgraph_view(G, filter_node=no_filter, filter_edge=no_filter):
    newG = freeze(G.__class__())
    newG._NODE_OK = filter_node
    newG._EDGE_OK = filter_edge
    newG._graph = G
    newG.graph = G.graph
    newG._node = FilterAtlas(G._node, filter_node)
    Adj = FilterAdjacency

    newG._adj = Adj(G._adj, filter_node, filter_edge)
    return newG
def generic_graph_view(G, create_using=None):
    if create_using is None:
        newG = G.__class__()
    else:
        newG = empty_graph(0, create_using)
    if G.is_multigraph() != newG.is_multigraph():
        raise NetworkXError("Multigraph for G must agree with create_using")
    newG = freeze(newG)

    newG._graph = G
    newG.graph = G.graph

    newG._node = G._node
    newG._adj = G._adj
    return newG
def to_networkx_graph(data, create_using=None, multigraph_input=False):

    if hasattr(data, "adj"):
        try:
            result = from_dict_of_dicts(data.adj,
                                        create_using=create_using,
                                        multigraph_input=data.is_multigraph())
            if hasattr(data, 'graph'):
                result.graph.update(data.graph)
            if hasattr(data, 'nodes'):
                result._node.update((n, dd.copy()) for n, dd in data.nodes.items())
            return result
        except:
            raise NetworkXError("Input is not a correct NetworkX graph.")
    if isinstance(data, dict):
        try:
            return from_dict_of_dicts(data, create_using=create_using,
                                      multigraph_input=multigraph_input)
        except:
            try:
                return from_dict_of_lists(data, create_using=create_using)
            except:
                raise TypeError("Input is not known type.")
    if (isinstance(data, (list, tuple)) or
            any(hasattr(data, attr) for attr in ['_adjdict', 'next', '__next__'])):
        try:
            return from_edgelist(data, create_using=create_using)
        except:
            raise NetworkXError("Input is not a valid edge list")
def to_dict_of_lists(G, nodelist=None):
    if nodelist is None:
        nodelist = G

    d = {}
    for n in nodelist:
        d[n] = [nbr for nbr in G.neighbors(n) if nbr in nodelist]
    return d
def from_dict_of_lists(d, create_using=None):
    G = empty_graph(0, create_using)
    G.add_nodes_from(d)
    G.add_edges_from(((node, nbr) for node, nbrlist in d.items()
                      for nbr in nbrlist))
    return G
def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    G = empty_graph(0, create_using)
    G.add_nodes_from(d)
    G.add_edges_from(((u, v, data)
                          for u, nbrs in d.items()
                          for v, data in nbrs.items()))
    return G
def from_edgelist(edgelist, create_using=None):
    G = empty_graph(0, create_using)
    G.add_edges_from(edgelist)
    return G
class Graph(object):
    node_dict_factory = dict
    adjlist_outer_dict_factory = dict
    adjlist_inner_dict_factory = dict
    edge_attr_dict_factory = dict

    def __init__(self, incoming_graph_data=None, **attr):
        self.node_dict_factory = ndf = self.node_dict_factory
        self.adjlist_outer_dict_factory = self.adjlist_outer_dict_factory
        self.adjlist_inner_dict_factory = self.adjlist_inner_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory

        self.graph = {}
        self._node = ndf()
        self._adj = self.adjlist_outer_dict_factory()

        if incoming_graph_data is not None:
            to_networkx_graph(incoming_graph_data, create_using=self)

        self.graph.update(attr)

    @property
    def adj(self):
        return AdjacencyView(self._adj)

    @property
    def name(self):
        return self.graph.get('name', '')

    @name.setter
    def name(self, s):
        self.graph['name'] = s

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self._node)

    def __contains__(self, n):
        try:
            return n in self._node
        except TypeError:
            return False

    def __len__(self):
        return len(self._node)

    def __getitem__(self, n):
        return self.adj[n]

    def add_node(self, node_for_adding, **attr):
        if node_for_adding not in self._node:
            self._adj[node_for_adding] = self.adjlist_inner_dict_factory()
            self._node[node_for_adding] = attr
        else:
            self._node[node_for_adding].update(attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        for n in nodes_for_adding:
            try:
                if n not in self._node:
                    self._adj[n] = self.adjlist_inner_dict_factory()
                    self._node[n] = attr.copy()
                else:
                    self._node[n].update(attr)
            except TypeError:
                nn, ndict = n
                if nn not in self._node:
                    self._adj[nn] = self.adjlist_inner_dict_factory()
                    newdict = attr.copy()
                    newdict.update(ndict)
                    self._node[nn] = newdict
                else:
                    olddict = self._node[nn]
                    olddict.update(attr)
                    olddict.update(ndict)

    def remove_node(self, n):
        adj = self._adj
        try:
            nbrs = list(adj[n])
            del self._node[n]
        except KeyError:
            raise NetworkXError("The node %s is not in the graph." % (n,))
        for u in nbrs:
            del adj[u][n]
        del adj[n]

    def remove_nodes_from(self, nodes):
        adj = self._adj
        for n in nodes:
            try:
                del self._node[n]
                for u in list(adj[n]):
                    del adj[u][n]
                del adj[n]
            except KeyError:
                pass

    @property
    def nodes(self):
        nodes = NodeView(self)
        self.__dict__['nodes'] = nodes
        return nodes
    node = nodes

    def number_of_nodes(self):
        return len(self._node)

    def order(self):
        return len(self._node)

    def has_node(self, n):
        try:
            return n in self._node
        except TypeError:
            return False

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        u, v = u_of_edge, v_of_edge

        if u not in self._node:
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = {}
        if v not in self._node:
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = {}

        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._adj[u][v] = datadict
        self._adj[v][u] = datadict

    def add_edges_from(self, ebunch_to_add, **attr):
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}
            else:
                raise NetworkXError(
                    "Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
            if u not in self._node:
                self._adj[u] = self.adjlist_inner_dict_factory()
                self._node[u] = {}
            if v not in self._node:
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = {}
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict

    def remove_edge(self, u, v):
        try:
            del self._adj[u][v]
            if u != v:
                del self._adj[v][u]
        except KeyError:
            raise NetworkXError("The edge %s-%s is not in the graph" % (u, v))

    def remove_edges_from(self, ebunch):
        adj = self._adj
        for e in ebunch:
            u, v = e[:2]
            if u in adj and v in adj[u]:
                del adj[u][v]
                if u != v:
                    del adj[v][u]

    def update(self, edges=None, nodes=None):
        if edges is not None:
            if nodes is not None:
                self.add_nodes_from(nodes)
                self.add_edges_from(edges)
            else:

                try:
                    graph_nodes = edges.nodes
                    graph_edges = edges.edges
                except AttributeError:

                    self.add_edges_from(edges)
                else:
                    self.add_nodes_from(graph_nodes.data())
                    self.add_edges_from(graph_edges.data())
                    self.graph.update(edges.graph)
        elif nodes is not None:
            self.add_nodes_from(nodes)
        else:
            raise NetworkXError("update needs nodes or edges input")

    def has_edge(self, u, v):
        try:
            return v in self._adj[u]
        except KeyError:
            return False

    def neighbors(self, n):
        try:
            return iter(self._adj[n])
        except KeyError:
            raise NetworkXError("The node %s is not in the graph." % (n,))

    @property
    def edges(self):
        return EdgeView(self)

    def get_edge_data(self, u, v, default=None):
        try:
            return self._adj[u][v]
        except KeyError:
            return default

    def adjacency(self):
        return iter(self._adj.items())

    @property
    def degree(self):
        return DegreeView(self)

    def clear(self):
        self._adj.clear()
        self._node.clear()
        self.graph.clear()

    def is_multigraph(self):
        return False

    def is_directed(self):
        return False

    def copy(self, as_view=False):
        if as_view is True:
            return generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from((n, d.copy()) for n, d in self._node.items())
        G.add_edges_from((u, v, datadict.copy())
                         for u, nbrs in self._adj.items()
                         for v, datadict in nbrs.items())
        return G

    def subgraph(self, nodes):
        induced_nodes = show_nodes(self.nbunch_iter(nodes))

        subgraph = subgraph_view
        if hasattr(self, '_NODE_OK'):
            return subgraph(self._graph, induced_nodes, self._EDGE_OK)
        return subgraph(self, induced_nodes)

    def edge_subgraph(self, edges):
        return edge_subgraph(self, edges)

    def size(self, weight=None):
        s = sum(d for v, d in self.degree(weight=weight))
        return s // 2 if weight is None else s / 2

    def number_of_edges(self, u=None, v=None):
        if u is None:
            return int(self.size())
        if v in self._adj[u]:
            return 1
        return 0

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:
            bunch = iter(self._adj)
        elif nbunch in self:
            bunch = iter([nbunch])
        else:
            def bunch_iter(nlist, adj):
                try:
                    for n in nlist:
                        if n in adj:
                            yield n
                except TypeError as e:
                    message = e.args[0]

                    if 'iter' in message:
                        msg = "nbunch is not a node or a sequence of nodes."
                        raise NetworkXError(msg)

                    elif 'hashable' in message:
                        msg = "Node {} in sequence nbunch is not a valid node."
                        raise NetworkXError(msg.format(n))
                    else:
                        raise
            bunch = bunch_iter(nbunch, self._adj)
        return bunch
def nodes_or_number(which_args):
    def _nodes_or_number(func_to_be_decorated, *args, **kw):
        try:
            iter_wa = iter(which_args)
        except TypeError:
            iter_wa = (which_args,)

        new_args = list(args)
        for i in iter_wa:
            n = args[i]
            try:
                nodes = list(range(n))
            except TypeError:
                nodes = tuple(n)
            else:
                if n < 0:
                    msg = "Negative number of nodes not valid: %i" % n
                    raise NetworkXError(msg)
            new_args[i] = (n, nodes)
        return func_to_be_decorated(*new_args, **kw)
    return _nodes_or_number
def empty_graph(n=0, create_using=None, default=Graph):
    if create_using is None:
        G = default()
    elif hasattr(create_using, '_adj'):

        create_using.clear()
        G = create_using
    else:

        G = create_using()

    n_name, nodes = n
    G.add_nodes_from(nodes)
    return G
import sys
def create_graph(n, m, k, file):
    G = Graph()
    for l in range(1, n+1):
        for v in range(1, m+1):
            G.add_node((l, v))

    G.add_node((0, 0))
    G.add_node((n+1, 0))
    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 0:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))

    for v in range(1, m+1):
        G.add_edge((0, 0), (1, v))
        G.add_edge((n+1, 0), (n, v))
    for l in range(1, n):
        for j in range(1, m+1):
            line = file.readline()
            numbers = line.split(" ")
            numbers = [int(x) for x in numbers if x != "\n"]
            for v in numbers:
                G.add_edge((l, j), (l+1, v))
    for l in range(1, n+1):
        for v in range(1, m+1):
            if v % 2 == 1:
                if v < m:
                    G.add_edge((l, v), (l, v+1))
                else:
                    G.add_edge((l, v), (l, 1))
    return G

def euler2(G, source):
    stack = [source]
    C = []
    while len(stack) > 0:
        current = stack[-1]
        if G.degree(current) == 0:
            stack.pop()
            C.append(current)
        else:
            _, n = arbitrary_element(G.edges(current))
            G.remove_edge(current, n)
            stack.append(n)
    ret = []
    C.reverse()
    prev = None
    for v in C:
        if prev is not None:
            ret.append((prev, v))
        prev = v
    return ret

def correct_v(v, l, player):
    return (v % 2 == l % 2) if player == 1 else (v % 2 != l % 2)

def weird_mod(v, m):
    if v > m:
        return 1
    if v < 1:
        return m
    return v

def direction(layer, v_layer, v, player):
    if player == 1:
        if layer % 2 == v_layer % 2:
            if v % 2 == 0:
                return 1
            return -1
        else:
            if v % 2 == 0:
                return -1
            return 1
    else:
        if layer % 2 == v_layer % 2:
            if v % 2 == 0:
                return -1
            return 1
        else:
            if v % 2 == 0:
                return 1
            return -1

def ceremony_tour(G, reduced, source, n, m, k, layer, player = 1):
    path = []

    current = None

    while current != source:
        if current is None:
            current = source
        next_vert = None#arbitrary_element(reduced.edges(current))
        l, v = current
        if l == layer:
            if (v == 0 and l == 0) or not correct_v(v, l, player):
                for e in reduced.edges(current):
                    _, v2 = e
                    l_next, v_next = v2
                    if l_next == layer + 1 and (correct_v(v_next, l_next, player) or v_next == 0):
                        next_vert = v2
                        break
            else:
                ideal_v = weird_mod(v + direction(layer, l, v, player), m)
                next_vert = (l, ideal_v)

        elif l == layer + 1:
            if (v == 0 and l == n+1) or not correct_v(v, l, player):
                for e in reduced.edges(current):
                    _, v2 = e
                    l_next, v_next = v2
                    if l_next == layer and (correct_v(v_next, l_next, player) or v_next == 0):
                        next_vert = v2
                        break
            else:
                ideal_v = weird_mod(v + direction(layer, l, v, player), m)
                next_vert = (l, ideal_v)
        if next_vert is None:
            _, next_vert = arbitrary_element(reduced.edges(current))
        edge = current, next_vert
        G.remove_edge(current, next_vert)
        reduced.remove_edge(current, next_vert)
        path.append(edge)
        current = next_vert

    return path

def reduce_graph(G, n, m, k):
    reduced = G.copy()

    for l in range(1, n):
        nodes = [(l, x) for x in range(1, m + 1)] + [(l + 1, x) for x in range(1, m + 1)]
        sub2 = G.subgraph(nodes)
        edges2 = {e for e in sub2.edges if e[0][0] != e[1][0]}
        sub2 = sub2.edge_subgraph(edges2)
        matching = find_perfect_matching(sub2, [(l, x) for x in range(1, m + 1)])
        for e in edges2:
            if e not in matching:
                reduced.remove_edge(e[0], e[1])
    return reduced


def solve_graph(G, reduced, n, m, k, player = 1, silver = []):
    full_path = []
    for l in range(0, n+1):
        if l == 0:
            while len(full_path) != 3 * (m / 2):
                full_path += ceremony_tour(G, reduced, (0, 0), n, m, k, l, player)
        else:
            new_full_path = full_path
            for i in range(len(full_path) - 1, -1, -1):
                edge = full_path[i]
                v1, v2 = edge
                if v1[0] == l and reduced.degree(v1) > 0:
                    path = ceremony_tour(G, reduced, v1, n, m, k, l, player)
                    new_full_path = new_full_path[:i] + path + new_full_path[i:]
            full_path = new_full_path

    new_full_path = full_path
    for i in range(len(full_path) - 1, -1, -1):
        edge = full_path[i]
        v1, v2 = edge
        if G.degree(v1) > 0:
            path = euler2(G, v1)
            new_full_path = new_full_path[:i] + path + new_full_path[i:]
    full_path = new_full_path
    return full_path, []

def find_perfect_matching(G, sets):
    matching = hopcroft_karp_matching(G, sets)
    return {(key, value) for key, value in matching.items()}

def read_and_solve_one(file):
    numbers = file.readline()
    numbers = numbers.rstrip()
    numbers = [int(x) for x in numbers.split(" ")]
    n, m, k = numbers
    G = create_graph(n, m, k, file)
    path, path2 = solve_graph_twice(G, n, m, k)
    print_paths(path, path2)

def solve_graph_twice(G, n, m, k):
    G2 = G.copy()
    reduced = reduce_graph(G, n, m, k)
    reduced2 = reduced.copy()
    path, silver = solve_graph(G, reduced, n, m, k)
    path2, _ = solve_graph(G2, reduced2, n, m, k, 2, silver)
    return (path, path2)

def print_paths(path, path2):
    res = ""
    for v1, v2 in path:
        res += "{} {}\n".format(v2[0], v2[1])
    for v1, v2 in path2:
        res += "{} {}\n".format(v2[0], v2[1])
    sys.stdout.write(res)

def read_and_solve(file):
    testcases = int(file.readline().rstrip())
    for t in range(0, testcases):
        print("Case {}: Yes".format(t+1))
        read_and_solve_one(file)

def main():
    filein = sys.stdin
    read_and_solve(filein)

if __name__ == "__main__":
    main()
