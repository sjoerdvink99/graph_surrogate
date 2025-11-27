import gzip
import bz2
import json
import mmap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx


@dataclass
class GraphMetadata:
    name: str
    num_nodes: int
    num_edges: int
    node_types: list[str]
    edge_types: list[str] = field(default_factory=list)
    attributes: dict[str, list] = field(default_factory=dict)
    source_format: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "attributes": self.attributes,
            "source_format": self.source_format,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphMetadata":
        return cls(**d)


def open_file(path: Path, mode: str = "r"):
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    elif suffix == ".bz2":
        return bz2.open(path, mode + "t", encoding="utf-8")
    else:
        return open(path, mode, encoding="utf-8")


def detect_format(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    elif name.endswith(".bz2"):
        name = name[:-4]

    suffix = Path(name).suffix.lower()

    if suffix == ".json":
        return "json"
    elif suffix == ".gml":
        return "gml"
    elif suffix in (".graphml", ".xml"):
        return "graphml"
    elif suffix in (".edgelist", ".edges", ".txt", ".tsv", ".csv"):
        return "edgelist"
    else:
        with open_file(path) as f:
            first_line = f.readline().strip()
            if first_line.startswith("{") or first_line.startswith("["):
                return "json"
            elif first_line.lower().startswith("graph"):
                return "gml"
            elif first_line.startswith("<?xml"):
                return "graphml"
            else:
                return "edgelist"


def load_json_graph(path: Path) -> nx.Graph:
    with open_file(path) as f:
        data = json.load(f)

    G = nx.Graph()

    if "nodes" in data:
        nodes = data.get("nodes", [])
        edges = data.get("edges", data.get("links", []))

        for node in nodes:
            # Handle [id, attrs] format (e.g., BRON dataset)
            if isinstance(node, list) and len(node) >= 2:
                node_id = node[0]
                attrs = node[1] if isinstance(node[1], dict) else {}
            else:
                # Handle {id: ..., ...} format
                node_id = node.get("id", node.get("_id"))
                attrs = {k: v for k, v in node.items() if k not in ("id", "_id")}

            if "datatype" in attrs and "node_type" not in attrs:
                attrs["node_type"] = attrs.pop("datatype")
            if "type" in attrs and "node_type" not in attrs:
                attrs["node_type"] = attrs.pop("type")
            if "node_type" not in attrs:
                attrs["node_type"] = "default"

            # Remove nested dicts that can't be used as node attributes
            attrs = {k: v for k, v in attrs.items() if not isinstance(v, dict)}

            G.add_node(node_id, **attrs)

        for edge in edges:
            # Handle [src, tgt] or [src, tgt, attrs] format
            if isinstance(edge, list) and len(edge) >= 2:
                src, tgt = edge[0], edge[1]
                attrs = edge[2] if len(edge) > 2 and isinstance(edge[2], dict) else {}
            else:
                # Handle {source: ..., target: ...} format
                src = edge.get("source", edge.get("_from", edge.get("src")))
                tgt = edge.get("target", edge.get("_to", edge.get("dst")))
                attrs = {k: v for k, v in edge.items()
                         if k not in ("source", "target", "_from", "_to", "src", "dst")}

            if src is not None and tgt is not None:
                if G.has_node(src) and G.has_node(tgt):
                    G.add_edge(src, tgt, **attrs)

    elif isinstance(data, dict) and all(isinstance(v, (list, dict)) for v in data.values()):
        for node, neighbors in data.items():
            if not G.has_node(node):
                G.add_node(node, node_type="default")
            if isinstance(neighbors, list):
                for neighbor in neighbors:
                    if not G.has_node(neighbor):
                        G.add_node(neighbor, node_type="default")
                    G.add_edge(node, neighbor)
            elif isinstance(neighbors, dict):
                for neighbor, attrs in neighbors.items():
                    if not G.has_node(neighbor):
                        G.add_node(neighbor, node_type="default")
                    G.add_edge(node, neighbor, **(attrs if isinstance(attrs, dict) else {}))

    return G


def load_edgelist_graph(path: Path, delimiter: Optional[str] = None) -> nx.Graph:
    G = nx.Graph()

    with open_file(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            if delimiter is None:
                if "\t" in line:
                    delimiter = "\t"
                elif "," in line:
                    delimiter = ","
                else:
                    delimiter = None

            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()

            if len(parts) < 2:
                continue

            src, dst = parts[0], parts[1]

            if not G.has_node(src):
                G.add_node(src, node_type="default")
            if not G.has_node(dst):
                G.add_node(dst, node_type="default")

            if len(parts) >= 3:
                try:
                    weight = float(parts[2])
                    G.add_edge(src, dst, weight=weight)
                except ValueError:
                    G.add_edge(src, dst)
            else:
                G.add_edge(src, dst)

    return G


def load_gml_graph(path: Path) -> nx.Graph:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            G = nx.read_gml(f)
    elif str(path).endswith(".bz2"):
        with bz2.open(path, "rt") as f:
            G = nx.read_gml(f)
    else:
        G = nx.read_gml(str(path))

    for node in G.nodes():
        if "node_type" not in G.nodes[node]:
            G.nodes[node]["node_type"] = G.nodes[node].get("label", "default")

    return G


def load_graphml_graph(path: Path) -> nx.Graph:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            G = nx.read_graphml(f)
    elif str(path).endswith(".bz2"):
        with bz2.open(path, "rt") as f:
            G = nx.read_graphml(f)
    else:
        G = nx.read_graphml(str(path))

    for node in G.nodes():
        if "node_type" not in G.nodes[node]:
            G.nodes[node]["node_type"] = "default"

    return G


def add_degree_bins(G: nx.Graph, bins: list[int] = [2, 5, 10, 20, 50]) -> None:
    for node in G.nodes():
        degree = G.degree(node)
        degree_bin = len(bins)
        for i, threshold in enumerate(bins):
            if degree <= threshold:
                degree_bin = i
                break
        G.nodes[node]["degree_bin"] = degree_bin


def extract_metadata(G: nx.Graph, name: str = "graph", source_format: str = "unknown") -> GraphMetadata:
    node_types = set()
    edge_types = set()
    attributes: dict[str, set] = {}

    for _, data in G.nodes(data=True):
        node_types.add(data.get("node_type", "default"))
        for key, val in data.items():
            if key not in ("node_type", "degree_bin") and val is not None:
                if key not in attributes:
                    attributes[key] = set()
                attributes[key].add(str(val))

    for _, _, data in G.edges(data=True):
        if "type" in data:
            edge_types.add(data["type"])
        elif "label" in data:
            edge_types.add(data["label"])

    attr_dict = {k: sorted(v) for k, v in attributes.items()}

    return GraphMetadata(
        name=name,
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        node_types=sorted(node_types),
        edge_types=sorted(edge_types),
        attributes=attr_dict,
        source_format=source_format,
    )


def load_graph(
    path: str | Path,
    name: Optional[str] = None,
    add_degree_bin: bool = True,
) -> tuple[nx.Graph, GraphMetadata]:
    path = Path(path)
    if name is None:
        name = path.stem
        if name.endswith(".gz") or name.endswith(".bz2"):
            name = Path(name).stem

    fmt = detect_format(path)

    print(f"Loading graph from {path} (format: {fmt})...")

    if fmt == "json":
        G = load_json_graph(path)
    elif fmt == "edgelist":
        G = load_edgelist_graph(path)
    elif fmt == "gml":
        G = load_gml_graph(path)
    elif fmt == "graphml":
        G = load_graphml_graph(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if add_degree_bin:
        add_degree_bins(G)

    metadata = extract_metadata(G, name=name, source_format=fmt)

    print(f"Loaded: {metadata.num_nodes:,} nodes, {metadata.num_edges:,} edges")
    print(f"Node types: {metadata.node_types}")

    return G, metadata


def save_graph(G: nx.Graph, path: str | Path, metadata: Optional[GraphMetadata] = None) -> None:
    path = Path(path)
    nx.write_gml(G, str(path))

    if metadata:
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)


class StreamingEdgeIterator:
    def __init__(self, path: str | Path, batch_size: int = 10000):
        self.path = Path(path)
        self.batch_size = batch_size
        self.format = detect_format(self.path)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        if self.format == "edgelist":
            yield from self._iter_edgelist()
        elif self.format == "json":
            yield from self._iter_json()
        else:
            raise NotImplementedError(f"Streaming not supported for {self.format}")

    def _iter_edgelist(self) -> Iterator[tuple[str, str]]:
        with open_file(self.path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    yield parts[0], parts[1]

    def _iter_json(self) -> Iterator[tuple[str, str]]:
        with open_file(self.path) as f:
            data = json.load(f)

        edges = data.get("edges", data.get("links", []))
        for edge in edges:
            src = edge.get("source", edge.get("_from"))
            dst = edge.get("target", edge.get("_to"))
            if src and dst:
                yield src, dst


def get_node_sample(
    path: str | Path,
    n: int = 1000,
    seed: int = 42,
) -> list[str]:
    import random
    random.seed(seed)

    path = Path(path)
    fmt = detect_format(path)

    nodes = set()

    if fmt == "edgelist":
        with open_file(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    nodes.add(parts[0])
                    nodes.add(parts[1])
    elif fmt == "json":
        with open_file(path) as f:
            data = json.load(f)
        if "nodes" in data:
            nodes = {n.get("id", n.get("_id")) for n in data["nodes"]}
        else:
            edges = data.get("edges", data.get("links", []))
            for edge in edges:
                nodes.add(edge.get("source", edge.get("_from")))
                nodes.add(edge.get("target", edge.get("_to")))

    nodes_list = list(nodes)
    if len(nodes_list) <= n:
        return nodes_list

    return random.sample(nodes_list, n)
