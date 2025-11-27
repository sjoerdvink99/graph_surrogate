import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from training.query_sampler import AttributeFilter, Query, QueryType


@dataclass
class EncoderConfig:
    node_types: list[str]
    attribute_names: list[str]
    attribute_values: dict[str, list]
    degree_bins: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    radii: list[int] = field(default_factory=lambda: [1, 2, 3])
    max_hops_options: list[int] = field(default_factory=lambda: [3, 4, 5, 6])

    @property
    def input_dim(self) -> int:
        total_attr_values = sum(len(v) for v in self.attribute_values.values())

        return (
            len(self.node_types)
            + len(self.degree_bins)
            + len(self.radii)
            + len(self.attribute_names) + 1
            + total_attr_values
            + len(self.node_types)
            + len(self.max_hops_options)
            + 2
        )

    def to_dict(self) -> dict:
        return {
            "node_types": self.node_types,
            "attribute_names": self.attribute_names,
            "attribute_values": self.attribute_values,
            "degree_bins": self.degree_bins,
            "radii": self.radii,
            "max_hops_options": self.max_hops_options,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        # Handle backwards compatibility with older configs
        attribute_values = d.get("attribute_values", {})
        if not attribute_values:
            # Reconstruct attribute_values from old-style separate fields
            attribute_values = {}
            for attr_name in d.get("attribute_names", []):
                # Map common attribute names to their value fields
                if attr_name == "severity" and "severities" in d:
                    attribute_values[attr_name] = d["severities"]
                elif attr_name == "department" and "departments" in d:
                    attribute_values[attr_name] = d["departments"]
                elif attr_name == "node_type":
                    attribute_values[attr_name] = d.get("node_types", [])
                elif attr_name == "degree_bin":
                    attribute_values[attr_name] = d.get("degree_bins", [0, 1, 2, 3, 4])
                elif attr_name == "motif_type" and "motif_types" in d:
                    attribute_values[attr_name] = d["motif_types"]
                else:
                    attribute_values[attr_name] = []

        return cls(
            node_types=d["node_types"],
            attribute_names=d["attribute_names"],
            attribute_values=attribute_values,
            degree_bins=d.get("degree_bins", [0, 1, 2, 3, 4]),
            radii=d.get("radii", [1, 2, 3]),
            max_hops_options=d.get("max_hops_options", [3, 4, 5, 6]),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EncoderConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class QueryEncoder:
    def __init__(self, config: EncoderConfig):
        self.config = config

        self.node_type_to_idx = {t: i for i, t in enumerate(config.node_types)}
        self.degree_bin_to_idx = {b: i for i, b in enumerate(config.degree_bins)}
        self.radius_to_idx = {r: i for i, r in enumerate(config.radii)}
        self.attr_name_to_idx = {a: i for i, a in enumerate(config.attribute_names)}
        self.max_hops_to_idx = {h: i for i, h in enumerate(config.max_hops_options)}

        self.attr_value_to_idx = {}
        offset = 0
        for attr_name in config.attribute_names:
            values = config.attribute_values.get(attr_name, [])
            for i, val in enumerate(values):
                self.attr_value_to_idx[(attr_name, val)] = offset + i
            offset += len(values)
        self.total_attr_values = offset

    def encode(self, query: Query) -> torch.Tensor:
        parts = []

        node_type_vec = torch.zeros(len(self.config.node_types))
        if query.start_node_type in self.node_type_to_idx:
            node_type_vec[self.node_type_to_idx[query.start_node_type]] = 1.0
        parts.append(node_type_vec)

        degree_vec = torch.zeros(len(self.config.degree_bins))
        if query.start_degree_bin in self.degree_bin_to_idx:
            degree_vec[self.degree_bin_to_idx[query.start_degree_bin]] = 1.0
        parts.append(degree_vec)

        radius_vec = torch.zeros(len(self.config.radii))
        if query.query_type == QueryType.COUNT and query.radius in self.radius_to_idx:
            radius_vec[self.radius_to_idx[query.radius]] = 1.0
        parts.append(radius_vec)

        attr_name_vec = torch.zeros(len(self.config.attribute_names) + 1)
        if query.query_type == QueryType.COUNT:
            if query.attribute_filter.name is None:
                attr_name_vec[-1] = 1.0
            elif query.attribute_filter.name in self.attr_name_to_idx:
                attr_name_vec[self.attr_name_to_idx[query.attribute_filter.name]] = 1.0
        else:
            attr_name_vec[-1] = 1.0
        parts.append(attr_name_vec)

        attr_value_vec = torch.zeros(self.total_attr_values)
        if query.query_type == QueryType.COUNT and query.attribute_filter.name is not None:
            key = (query.attribute_filter.name, query.attribute_filter.value)
            if key in self.attr_value_to_idx:
                attr_value_vec[self.attr_value_to_idx[key]] = 1.0
        parts.append(attr_value_vec)

        target_type_vec = torch.zeros(len(self.config.node_types))
        if query.query_type == QueryType.DISTANCE and query.target_type in self.node_type_to_idx:
            target_type_vec[self.node_type_to_idx[query.target_type]] = 1.0
        parts.append(target_type_vec)

        max_hops_vec = torch.zeros(len(self.config.max_hops_options))
        if query.query_type == QueryType.DISTANCE and query.max_hops in self.max_hops_to_idx:
            max_hops_vec[self.max_hops_to_idx[query.max_hops]] = 1.0
        parts.append(max_hops_vec)

        query_type_vec = torch.zeros(2)
        query_type_vec[0 if query.query_type == QueryType.COUNT else 1] = 1.0
        parts.append(query_type_vec)

        return torch.cat(parts)

    def encode_batch(self, queries: list[Query]) -> torch.Tensor:
        return torch.stack([self.encode(q) for q in queries])

    def encode_from_dict(self, query_dict: dict) -> torch.Tensor:
        attr_filter = AttributeFilter()
        if query_dict.get("attribute_filter"):
            af = query_dict["attribute_filter"]
            attr_filter = AttributeFilter(name=af.get("name"), value=af.get("value"))

        query_type = QueryType(query_dict.get("query_type", "count"))

        query = Query(
            query_type=query_type,
            start_node_id=query_dict.get("start_node_id", 0),
            start_node_type=query_dict.get("start_node_type", "Host"),
            start_degree_bin=query_dict.get("start_degree_bin", 0),
            radius=query_dict.get("radius", 1),
            attribute_filter=attr_filter,
            target_type=query_dict.get("target_type"),
            max_hops=query_dict.get("max_hops", 6),
        )
        return self.encode(query)
