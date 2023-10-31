import collections
import os
from typing import List, Tuple, Iterator, Optional, Any
from dataclasses import dataclass, field, asdict
import re
import json
from typing import List, Optional, Iterable


class DAG:
    def __init__(self, vertices: Optional[List[str]] = None,
                 edges: Optional[List[Tuple[str, str]]] = None,
                 labels: Optional[List[Optional[str]]] = None,
                 objects: Optional[List[Any]] = None):
        self._vertices = []
        self._edges = []
        self._v_labels = {}
        self._v_objects = {}
        self._v_set = set()
        self._pred = collections.defaultdict(set)
        self._succ = collections.defaultdict(set)
        self._all_pred = collections.defaultdict(set)
        self._all_succ = collections.defaultdict(set)

        if vertices is not None:
            if labels is not None:
                assert len(vertices) == len(labels)
            for i, v in enumerate(vertices):
                self.add_vertex(v, label=labels[i] if labels else None,
                                object=objects[i] if objects else None)

        if edges is not None:
            for sv, tv in edges:
                self.add_edge(sv, tv)

    def __str__(self):
        isolated_v = [v for v in self if not self.succ[v] and not self.pred[v]]
        res = '  '.join(isolated_v)
        if len(self.edges) > 0:
            res += '  ' + '  '.join(f'{sv}->{tv}' for sv, tv in self.edges)
        return '{ ' + res + ' }' if res else '{}'

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for v in self._vertices:
            yield v

    def __len__(self):
        return len(self._vertices)

    def __contains__(self, item):
        return item in self._v_set

    def __eq__(self, other):
        """Note: __eq__ does not test labels and objects"""
        if not (all(v in other for v in self) and all(v in self for v in other)):
            return False
        if not all(tv in other.succ[sv] for sv, tv in self.edges):
            return False
        if not all(tv in self.succ[sv] for sv, tv in other.edges):
            return False
        return True

    def get_label(self, v: str):
        return self._v_labels[v]

    def get_object(self, v: str):
        return self._v_objects[v]

    @property
    def edges(self):
        return self._edges

    @property
    def succ(self):
        return self._succ

    @property
    def pred(self):
        return self._pred

    @property
    def all_succ(self):
        return self._all_succ

    @property
    def all_pred(self):
        return self._all_pred

    def all_paths(self, sv: str, tv: str) -> Iterator[Tuple[str]]:
        def dfs(v):
            curr_path.append(v)
            if v == tv:
                yield tuple(curr_path)
            else:
                for v1 in self._succ[v]:
                    yield from dfs(v1)
            curr_path.pop(-1)

        curr_path = []
        return dfs(sv)

    def add_vertex(self, v: str, label: Optional[str] = None, object: Optional[Any] = None):
        assert isinstance(v, str)
        if v in self._v_set:
            raise ValueError(f'Vertex {v} already in graph')
        self._vertices.append(v)
        self._v_set.add(v)
        self._v_labels[v] = label
        self._v_objects[v] = object

    def add_edge(self, sv: str, tv: str):
        for v in (sv, tv):
            if v not in self._v_set:
                raise ValueError(f'Vertex {v} not in graph')
        if tv == sv:
            raise ValueError(f'Cannot add self-loop {sv} -> {tv}')
        if tv in self._succ[sv]:
            raise ValueError(f'Edge {sv} -> {tv} already in graph')
        if sv in self._all_succ[tv]:
            path = next(self.all_paths(tv, sv)) + (tv,)
            path = ' -> '.join(path)
            raise ValueError(f'Adding edge {sv} -> {tv} will create cycle {path}')

        self._edges.append((sv, tv))
        self._succ[sv].add(tv)
        self._pred[tv].add(sv)
        self._all_succ[sv] |= self._all_succ[tv] | {tv}
        self._all_pred[tv] |= self._all_pred[sv] | {sv}
        for v in self._all_pred[sv] | {sv}:
            self._all_succ[v] |= self._all_succ[sv]
        for v in self._all_succ[tv] | {tv}:
            self._all_pred[v] |= self._all_pred[tv]

    def topsort(self) -> List[str]:
        out_degree = {v: len(self._succ[v]) for v in self._vertices}
        roots = [v for v in self._vertices if out_degree[v] == 0]
        res = []
        while len(roots) > 0:
            v = roots.pop(0)
            res.append(v)
            for v1 in self._pred[v]:
                out_degree[v1] -= 1
                if out_degree[v1] == 0:
                    roots.append(v1)
        return res

    def lowest_common_ancestors(self, vertices: List[str]) -> List[str]:
        ancestors = self.all_pred[vertices[0]] | {vertices[0]}
        for v in vertices[1:]:
            ancestors &= self.all_pred[v] | {v}
        return [v for v in ancestors if len(self.all_succ[v] & ancestors) == 0]

    def clone(self) -> 'DAG':
        return self._subgraph(list(self))

    def transitive_closure(self) -> 'DAG':
        g = self.clone()
        for v in self:
            for v1 in self.all_succ[v] - self.succ[v]:
                g.add_edge(v, v1)
        return g

    def transitive_reduction(self) -> 'DAG':
        g = self.__class__()
        for v in self:
            g.add_vertex(v, label=self.get_label(v), object=self.get_object(v))
        for v in self:
            reachable = set()
            for v1 in self.succ[v]:
                reachable |= self.all_succ[v1]
            for v2 in self.succ[v] - reachable:
                g.add_edge(v, v2)
        return g

    def _subgraph(self, vertices: List[str]) -> 'DAG':
        g = self.__class__()
        for v in vertices:
            if v not in self:
                raise ValueError(f'Vertex {v} not in the graph')
            g.add_vertex(v, label=self.get_label(v), object=self.get_object(v))
        for sv, tv in self._edges:
            if sv in g._v_set and tv in g._v_set:
                g.add_edge(sv, tv)
        return g

    def subgraph(self, vertices: List[str]) -> 'DAG':
        return self._subgraph(vertices)

    def __sub__(self, other):
        g = self.__class__()
        for v in self:
            if v not in other:
                g.add_vertex(v, label=self.get_label(v), object=self.get_object(v))
        for sv, tv in self._edges:
            if sv in g and tv in g:
                g.add_edge(sv, tv)
        return g

    def __or__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        raise NotImplementedError()

    def _v_max_depths(self):
        def dfs(v, depth):
            v_max_depths[v] = max(v_max_depths[v], depth)
            for v1 in self.succ[v]:
                dfs(v1, depth + 1)

        v_max_depths = collections.defaultdict(int)
        for v in self:
            if len(self.pred[v]) == 0:
                dfs(v, 0)
        return v_max_depths

    def _maximum_spanning_edges(self):
        v_max_depths = self._v_max_depths()
        return [(max(list(self.pred[v]), key=lambda v: v_max_depths[v]), v)
                for v in self if len(self.pred[v]) > 0]

    @staticmethod
    def _split_to_multilines(s, maxlen=20):
        tokens = s.split()
        res = []
        curr_line_tokens = []
        curr_line_len = -1
        for t in tokens:
            if curr_line_len >= maxlen:
                res.append(' '.join(curr_line_tokens))
                curr_line_tokens = []
                curr_line_len = -1
            curr_line_tokens.append(t)
            curr_line_len += len(t) + 1
        if len(curr_line_tokens):
            res.append(' '.join(curr_line_tokens))
        return '\n'.join(res)

    def graphviz(self, title: Optional[str] = None, dpi: int = 96, transparent=False,
                 theme: str = 'light'):
        import graphviz

        g = graphviz.Digraph('G')
        g.attr(label=title, dpi=str(dpi),
               rankdir='LR', outputorder='edgesfirst', splines='line',
               compound='true', fontname='Sans Not-Rotated', fontsize='16',
               labelloc='t', labeljust='l', newrank='true',
               bgcolor='transparent' if transparent else '', )

        mst_edges = set(self._maximum_spanning_edges())

        node_attrs = dict(color='#20222a' if theme == 'dark' else '#222222',
                          fontcolor='#f3f3f3' if theme == 'dark' else '#222222',
                          fillcolor='#20222a' if theme == 'dark' else '#eeeeee',
                          shape='rect', height='0.3', margin='0.22,0.055',
                          fontsize='8', fontname='Sans Not-Rotated', style='rounded,filled,bold')
        for v in self._vertices:
            label = v
            if self._v_labels[v]:
                label += '\n' + '\n'.join([self._split_to_multilines(line) for line in self._v_labels[v].split('\n')])
            g.node(v, label=label, **node_attrs)

        edge_attrs = dict(arrowsize='0.5', penwidth='1.5', arrowhead='dot',
                          color='#20222a' if theme == 'dark' else '#222222',
                          tailport='e', headport='w')
        for sv, tv in self.edges:
            g.edge(sv, tv, constraint='true' if (sv, tv) in mst_edges else 'false', **edge_attrs)
        return g

    def draw(self, path: str, title: str = ''):
        assert path[-4:] in ('.pdf', '.png', '.jpg')
        output_format = path[-3:]
        dot_file_path = path[:-4]
        g = self.graphviz(title=title, dpi=96 if output_format == 'pdf' else 300)
        g.render(dot_file_path, format=output_format)
        os.remove(dot_file_path)


@dataclass
class WikidataConcept:
    wd_qid: str
    wd_label: str
    ufet_labels: List[str] = field(default_factory=list)
    description: Optional[str] = None
    alias_qids: List[str] = field(default_factory=list)

    def __post_init__(self):
        # note: this is a hack to make sure that the UFET types do not contain underscores
        self.ufet_labels = [t.replace('_', ' ') for t in self.ufet_labels]

    def __hash__(self):
        return hash(self.wd_qid)

    def __eq__(self, other):
        return other is not None and self.wd_qid == other.wd_qid

    def __str__(self):
        return f'Concept({self.wd_qid}, {self.wd_label})'

    def __repr__(self):
        return self.__str__()


class WikidataDAGOntology(DAG):
    default_root = WikidataConcept(
        wd_qid='Q35120',
        wd_label='entity',
        ufet_labels=['entity'],
        description='anything that can be considered, discussed, or observed'
    )

    def __init__(self):
        super().__init__()

    def concepts(self) -> Iterable[WikidataConcept]:
        for wd_qid in self:
            yield self.get_concept(wd_qid)

    def get_concept(self, wd_qid: str) -> WikidataConcept:
        return self.get_object(wd_qid)

    def add_concept(self, concept: WikidataConcept):
        label = f'{concept.wd_label} (Wikidata)'
        if len(concept.ufet_labels) > 0:
            label += f'\n{"/".join(concept.ufet_labels)} (UFET)'
        self.add_vertex(concept.wd_qid, label=label, object=concept)

    def add_relation(self, hypernym: WikidataConcept, hyponym: WikidataConcept):
        self.add_edge(hypernym.wd_qid, hyponym.wd_qid)

    def parents(self, concept: WikidataConcept) -> List[WikidataConcept]:
        return [self.get_object(v) for v in self.pred[concept.wd_qid]]

    def children(self, concept: WikidataConcept) -> List[WikidataConcept]:
        return [self.get_object(v) for v in self.succ[concept.wd_qid]]

    def all_parents(self, concept: WikidataConcept) -> List[WikidataConcept]:
        return [self.get_object(v) for v in self.all_pred[concept.wd_qid]]

    def all_children(self, concept: WikidataConcept) -> List[WikidataConcept]:
        return [self.get_object(v) for v in self.all_succ[concept.wd_qid]]

    def add_root(self, root: Optional[WikidataConcept] = None):
        if root is None:
            root = self.default_root
        if root.wd_qid not in self:
            self.add_concept(root)
        for c in self.roots():
            if c.wd_qid != root.wd_qid:
                self.add_relation(root, c)

    def roots(self) -> List[WikidataConcept]:
        return [self.get_object(v) for v in self if len(self.all_pred[v]) == 0]

    def subgraph_with_root(self, root: WikidataConcept) -> 'DAGOntology':
        return self._subgraph([root.wd_qid] + list(self.all_succ[root.wd_qid]))

    def subgraph(self, concepts: List[WikidataConcept]) -> 'DAGOntology':
        return self._subgraph([c.wd_qid for c in concepts])

    def lowest_common_ancestors(self, concepts: List[WikidataConcept]) -> List[WikidataConcept]:
        return [self.get_concept(qid) for qid in super().lowest_common_ancestors([c.wd_qid for c in concepts])]

    def save(self, output_path):
        dic = {
            'concepts': [asdict(self.get_object(v)) for v in self],
            'relations': [({'wd_qid': hype,
                            'wd_label': self.get_object(hype).wd_label},
                           {'wd_qid': hypo,
                            'wd_label': self.get_object(hypo).wd_label}) for hype, hypo in self.edges]
        }
        with open(output_path, 'w') as fout:
            json.dump(dic, fout, indent=2)

    @classmethod
    def from_file(cls, input_path):
        with open(input_path, 'r') as fin:
            dic = json.load(fin)
        ontology = cls()
        concept_dict = {v_dic['wd_qid']: WikidataConcept(**v_dic) for v_dic in dic['concepts']}
        for v_dic in dic['concepts']:
            ontology.add_concept(concept_dict[v_dic['wd_qid']])
        for hype_dic, hypo_dic in dic['relations']:
            ontology.add_relation(concept_dict[hype_dic['wd_qid']],
                                  concept_dict[hypo_dic['wd_qid']])
        return ontology


def load_concepts_from_ufet_mapping(ufet_mapping_path):
    res = []
    concept_dict = {}
    with open(ufet_mapping_path, 'r') as fin:
        for line in fin:
            ufet_label, wd_qid, wd_label, wd_desc, _ = line.split(',', 4)
            if wd_qid == 'null':
                continue
            assert re.match(r'Q[0-9]+', wd_qid), line
            if wd_desc.startswith('"') and wd_desc.endswith('"'):
                wd_desc = wd_desc[1:-1]
            if wd_qid in concept_dict:
                concept_dict[wd_qid].ufet_labels.append(ufet_label)
            else:
                c = WikidataConcept(wd_qid, wd_label,
                                    ufet_labels=[ufet_label],
                                    description=wd_desc)
                res.append(c)
                concept_dict[wd_qid] = c
    return res
