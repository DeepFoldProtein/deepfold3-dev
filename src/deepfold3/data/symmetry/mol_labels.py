"""Covalent-component symmetry labelling for the chain permutation (OF3 parity).

AF3 §4.2 treats a **covalently-connected component** — a polymer chain plus its
covalently-bonded glycans / ligands — as a single unit for the chain-permutation
alignment. Two glycosylated copies of the same protein must permute as whole
molecules (``{protein, glycan}`` ↔ ``{protein', glycan'}``), never as an
independent protein swap crossed with an independent glycan swap.

DeepFold3's chain permutation historically grouped chains strictly by
``entity_id`` (one chain = one permutation unit). This module adds the missing
component layer, mirroring OpenFold-3's
``core/data/primitives/permutation/mol_labels.py``:

* :func:`connected_chain_components` — union-find over the inter-chain covalent
  bonds → the connected set of chains that forms one molecule
  (OF3 ``chain_connected_molecule_iter`` / ``get_precursor_mol_groups``).
* :func:`assign_component_symmetry_classes` — a component-level symmetry class id
  assigned by **graph isomorphism** between coarse molecule graphs
  (OF3 ``construct_coarse_molecule_graph`` + ``assign_mol_symmetry_ids``): each
  chain is a node labelled by its ``entity_id`` (DeepFold3's sequence-identity
  key), each inter-chain bond is a labelled link-node pair (relative bonded-atom
  index, when known), and two components share a class iff their graphs are
  isomorphic under node labels. Isomorphism uses networkx ``vf2pp`` when
  available (exactly as OF3) and falls back to a hand-rolled labelled-graph
  backtracking matcher otherwise, so the module has no hard dependency on the
  (undeclared) networkx package.

The common case — no inter-chain covalent bonds — makes every chain its own
singleton component, so a component class reduces to an ``entity_id`` class and
the downstream permutation is behaviourally identical to the entity-only path.

Clean-room implementation; no OF3 code is reproduced.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass

# A single inter-chain bond, in either of two forms:
#   (asym_a, asym_b)                       — connectivity only (token path)
#   (asym_a, asym_b, rel_atom_a, rel_atom_b) — with per-endpoint relative atom
#                                              index (flat-atom path; OF3 fidelity)
Bond = tuple


# ---------------------------------------------------------------------------
# Connected components over inter-chain covalent bonds
# ---------------------------------------------------------------------------


def connected_chain_components(
    order: list[int],
    inter_chain_bonds: list[Bond] | None,
) -> tuple[list[tuple[int, ...]], dict[int, int]]:
    """Group chains into covalently-connected components (OF3 molecules).

    Union-find over the inter-chain bonds: chains joined (directly or
    transitively) by a covalent bond land in one component. Chains that appear
    in no inter-chain bond stay singletons — reproducing the historical
    one-chain-per-unit behaviour when ``inter_chain_bonds`` is empty / ``None``.

    Args:
        order: Chain ``asym_id``\\ s, in the caller's iteration order.
        inter_chain_bonds: Covalent bonds crossing a chain boundary. Only the
            first two entries (``asym_a, asym_b``) are read here. Bonds
            referencing an ``asym_id`` outside ``order`` — or self-bonds — are
            ignored.

    Returns:
        ``(components, comp_of)`` where ``components`` is a list of tuples of
        ``asym_id`` (each a covalently-connected set, members in
        first-appearance order; components ordered by first appearance of any
        member), and ``comp_of`` maps ``asym_id`` → component index.
    """
    parent: dict[int, int] = {a: a for a in order}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    order_set = set(order)
    if inter_chain_bonds:
        for bond in inter_chain_bonds:
            a, b = int(bond[0]), int(bond[1])
            if a != b and a in order_set and b in order_set:
                union(a, b)

    root_to_members: dict[int, list[int]] = defaultdict(list)
    for a in order:
        root_to_members[find(a)].append(a)

    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    comp_of: dict[int, int] = {}
    for a in order:
        r = find(a)
        if r in seen:
            continue
        seen.add(r)
        members = tuple(root_to_members[r])
        ci = len(components)
        components.append(members)
        for m in members:
            comp_of[m] = ci
    return components, comp_of


# ---------------------------------------------------------------------------
# Coarse molecule graph + labelled-graph isomorphism (OF3 parity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CoarseGraph:
    """Coarse molecule graph for one component (OF3 ``construct_coarse_molecule_graph``).

    Nodes are chain nodes (label ``("chain", entity_id)``) and, when relative
    bonded-atom indices are supplied, link nodes (label ``("link", rel_idx)``).
    ``edges`` holds undirected edges as ``frozenset`` pairs of node ids.
    """

    node_labels: dict[Hashable, Hashable]
    edges: frozenset

    def precursor_key(self) -> tuple:
        """Cheap isomorphism-invariant grouping key (OF3 ``PrecursorMolGroupID``).

        Isomorphic graphs must share (node count, edge count, sorted label
        multiset). Used to shard candidates before the exact match.
        """
        return (
            len(self.node_labels),
            len(self.edges),
            tuple(sorted(self.node_labels.values())),
        )


def _build_coarse_graph(
    members: tuple[int, ...],
    ent_of: dict[int, int],
    bonds: list[Bond],
) -> _CoarseGraph:
    """Build the coarse graph of a single covalently-connected component.

    Each chain becomes a node labelled by ``entity_id`` (DeepFold3's
    sequence-identity key — same entity ⇒ same residue/atom composition, so it
    subsumes OF3's per-chain atom/residue-name hash for grouping). Each
    inter-chain bond becomes either:

    * a labelled link-node pair (OF3-faithful) when relative bonded-atom indices
      are known — the link labels encode the attachment site, distinguishing a
      glycan bonded near a chain's N-terminus from one bonded elsewhere; or
    * a plain chain–chain edge when only connectivity is known (token path).
    """
    node_labels: dict[Hashable, Hashable] = {}
    edges: set = set()
    for a in members:
        node_labels[("c", a)] = ("chain", int(ent_of.get(a, -1)))
    for bond in bonds:
        a, b = int(bond[0]), int(bond[1])
        ra = int(bond[2]) if len(bond) >= 4 and bond[2] is not None else None
        rb = int(bond[3]) if len(bond) >= 4 and bond[3] is not None else None
        if ra is not None and rb is not None:
            la, lb = ("l", a, ra), ("l", b, rb)
            node_labels.setdefault(la, ("link", ra))
            node_labels.setdefault(lb, ("link", rb))
            edges.add(frozenset((("c", a), la)))
            edges.add(frozenset((("c", b), lb)))
            edges.add(frozenset((la, lb)))
        else:
            edges.add(frozenset((("c", a), ("c", b))))
    return _CoarseGraph(node_labels, frozenset(edges))


_NX: object = "unset"


def _networkx():
    """Return the imported networkx module, or ``None`` if unavailable (cached)."""
    global _NX
    if _NX == "unset":
        try:
            import networkx as nx

            _NX = nx
        except Exception:  # pragma: no cover - networkx nearly always present
            _NX = None
    return _NX


def _to_nx(nx, g: _CoarseGraph):
    graph = nx.Graph()
    for node, label in g.node_labels.items():
        graph.add_node(node, label=label)
    for edge in g.edges:
        u, v = tuple(edge)
        graph.add_edge(u, v)
    return graph


def _adjacency(g: _CoarseGraph) -> dict[Hashable, set]:
    adj: dict[Hashable, set] = {n: set() for n in g.node_labels}
    for edge in g.edges:
        u, v = tuple(edge)
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _iso_hand_rolled(g1: _CoarseGraph, g2: _CoarseGraph) -> bool:
    """Exact labelled-graph isomorphism by backtracking (networkx-free fallback).

    Correct for the tiny per-component graphs here (a handful of nodes). Nodes
    map only to same-label nodes; a partial map is extended only when it stays
    consistent with every already-mapped neighbour (VF2-style feasibility).
    """
    lab1, lab2 = g1.node_labels, g2.node_labels
    adj1, adj2 = _adjacency(g1), _adjacency(g2)

    by_label2: dict[Hashable, list] = defaultdict(list)
    for n in lab2:
        by_label2[lab2[n]].append(n)

    # Map the most-constrained (fewest candidates) source nodes first.
    order1 = sorted(lab1, key=lambda n: len(by_label2[lab1[n]]))
    mapping: dict[Hashable, Hashable] = {}
    used: set = set()

    def backtrack(i: int) -> bool:
        if i == len(order1):
            return True
        u = order1[i]
        for v in by_label2.get(lab1[u], ()):
            if v in used or len(adj1[u]) != len(adj2[v]):
                continue
            if all(mapping[w] in adj2[v] for w in adj1[u] if w in mapping):
                mapping[u] = v
                used.add(v)
                if backtrack(i + 1):
                    return True
                del mapping[u]
                used.discard(v)
        return False

    return backtrack(0)


def _graphs_isomorphic(g1: _CoarseGraph, g2: _CoarseGraph) -> bool:
    """Labelled-graph isomorphism test (networkx ``vf2pp`` preferred, exactly OF3)."""
    if g1.precursor_key() != g2.precursor_key():
        return False
    nx = _networkx()
    if nx is not None:
        from networkx.algorithms.isomorphism import vf2pp_isomorphism

        return vf2pp_isomorphism(_to_nx(nx, g1), _to_nx(nx, g2), node_label="label") is not None
    return _iso_hand_rolled(g1, g2)


def assign_component_symmetry_classes(
    components: list[tuple[int, ...]],
    ent_of: dict[int, int],
    inter_chain_bonds: list[Bond] | None,
) -> list[int]:
    """Assign a 1-based symmetry-class id per component (OF3 ``mol_entity_id``).

    Two components share a class iff their coarse molecule graphs are isomorphic
    under node labels — i.e. they are the same molecule up to relabelling of
    symmetric chains. Reduces to ``entity_id`` grouping for singleton
    components (a lone chain node is isomorphic to another iff their entity
    labels match).

    Args:
        components: Output of :func:`connected_chain_components`.
        ent_of: ``asym_id`` → ``entity_id``.
        inter_chain_bonds: The same bonds passed to
            :func:`connected_chain_components`; relative-atom entries (length-4
            tuples) sharpen the isomorphism to the attachment site.

    Returns:
        ``class_of[i]`` — the symmetry-class id of component ``i``.
    """
    comp_of: dict[int, int] = {}
    for ci, members in enumerate(components):
        for m in members:
            comp_of[m] = ci

    bonds_by_comp: dict[int, list[Bond]] = defaultdict(list)
    if inter_chain_bonds:
        for bond in inter_chain_bonds:
            a, b = int(bond[0]), int(bond[1])
            if a == b or a not in comp_of or b not in comp_of:
                continue
            bonds_by_comp[comp_of[a]].append(bond)

    graphs = [_build_coarse_graph(components[ci], ent_of, bonds_by_comp.get(ci, [])) for ci in range(len(components))]

    class_of = [0] * len(components)
    reps: dict[tuple, list[tuple[_CoarseGraph, int]]] = defaultdict(list)
    next_class = 1
    for i, g in enumerate(graphs):
        key = g.precursor_key()
        cid = None
        for rep_g, rep_cid in reps[key]:
            if _graphs_isomorphic(rep_g, g):
                cid = rep_cid
                break
        if cid is None:
            cid = next_class
            next_class += 1
        class_of[i] = cid
        reps[key].append((g, cid))
    return class_of
