import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from training.query_sampler import AttributeFilter, Query, QueryType


DEGREE_BINS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000]


def get_degree_bin(degree: int) -> int:
    for i, threshold in enumerate(DEGREE_BINS):
        if degree <= threshold:
            return i
    return len(DEGREE_BINS)


@dataclass
class EncoderConfig:
    node_types: list[str]
    attribute_names: list[str]
    attribute_values: dict[str, list]
    degree_bins: list[int] = field(default_factory=lambda: DEGREE_BINS.copy())
    radii: list[int] = field(default_factory=lambda: [1, 2, 3])
    max_hops_options: list[int] = field(default_factory=lambda: [3, 4, 5, 6])
    use_structural_features: bool = True
    num_structural_features: int = 8

    @property
    def input_dim(self) -> int:
        total_attr_values = sum(len(v) for v in self.attribute_values.values())
        return (
            len(self.node_types)
            + len(self.degree_bins) + 1
            + len(self.radii)
            + len(self.attribute_names) + 1
            + total_attr_values
            + len(self.node_types)
            + len(self.max_hops_options)
            + 2
        )

    @property
    def input_dim_with_structural(self) -> int:
        if self.use_structural_features:
            return self.input_dim + self.num_structural_features
        return self.input_dim

    def to_dict(self) -> dict:
        return {
            "node_types": self.node_types,
            "attribute_names": self.attribute_names,
            "attribute_values": self.attribute_values,
            "degree_bins": self.degree_bins,
            "radii": self.radii,
            "max_hops_options": self.max_hops_options,
            "use_structural_features": self.use_structural_features,
            "num_structural_features": self.num_structural_features,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        return cls(
            node_types=d["node_types"],
            attribute_names=d["attribute_names"],
            attribute_values=d.get("attribute_values", {}),
            degree_bins=d.get("degree_bins", DEGREE_BINS),
            radii=d.get("radii", [1, 2, 3]),
            max_hops_options=d.get("max_hops_options", [3, 4, 5, 6]),
            use_structural_features=d.get("use_structural_features", True),
            num_structural_features=d.get("num_structural_features", 8),
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

    def get_degree_bin(self, degree: int) -> int:
        for i, threshold in enumerate(self.config.degree_bins):
            if degree <= threshold:
                return i
        return len(self.config.degree_bins)

    def encode_indices(self, query: Query) -> dict[str, int]:
        indices = {}
        indices['node_type'] = self.node_type_to_idx.get(query.start_node_type, 0)
        indices['degree_bin'] = self.get_degree_bin(query.start_degree_bin)

        if query.query_type == QueryType.COUNT:
            indices['radius'] = self.radius_to_idx.get(query.radius, 0)
        else:
            indices['radius'] = len(self.config.radii)

        if query.query_type == QueryType.COUNT and query.attribute_filter.name is not None:
            indices['attr_name'] = self.attr_name_to_idx.get(
                query.attribute_filter.name, len(self.config.attribute_names)
            )
        else:
            indices['attr_name'] = len(self.config.attribute_names)

        if query.query_type == QueryType.COUNT and query.attribute_filter.name is not None:
            key = (query.attribute_filter.name, query.attribute_filter.value)
            indices['attr_value'] = self.attr_value_to_idx.get(key, self.total_attr_values)
        else:
            indices['attr_value'] = self.total_attr_values

        if query.query_type == QueryType.DISTANCE:
            indices['target_type'] = self.node_type_to_idx.get(query.target_type, 0)
        else:
            indices['target_type'] = len(self.config.node_types)

        if query.query_type == QueryType.DISTANCE:
            indices['max_hops'] = self.max_hops_to_idx.get(query.max_hops, 0)
        else:
            indices['max_hops'] = len(self.config.max_hops_options)

        indices['query_type'] = 0 if query.query_type == QueryType.COUNT else 1
        return indices

    def encode_indices_batch(self, queries: list[Query]) -> dict[str, torch.Tensor]:
        batch_indices = [self.encode_indices(q) for q in queries]
        return {
            key: torch.tensor([d[key] for d in batch_indices], dtype=torch.long)
            for key in batch_indices[0].keys()
        }

    def encode(self, query: Query) -> torch.Tensor:
        parts = []

        node_type_vec = torch.zeros(len(self.config.node_types))
        if query.start_node_type in self.node_type_to_idx:
            node_type_vec[self.node_type_to_idx[query.start_node_type]] = 1.0
        parts.append(node_type_vec)

        degree_vec = torch.zeros(len(self.config.degree_bins) + 1)
        bin_idx = self.get_degree_bin(query.start_degree_bin)
        degree_vec[bin_idx] = 1.0
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


def create_encoder_config(G, node_types: list[str], metadata) -> EncoderConfig:
    attr_values = {}
    for _, data in G.nodes(data=True):
        for key, val in data.items():
            if key not in ("node_type", "degree_bin") and val is not None:
                if key not in attr_values:
                    attr_values[key] = set()
                attr_values[key].add(val)

    attr_values["node_type"] = set(node_types)

    attribute_values = {}
    for k, v in attr_values.items():
        if len(v) <= 100:
            attribute_values[k] = sorted(v, key=str)

    attribute_names = sorted(attribute_values.keys())

    return EncoderConfig(
        node_types=node_types,
        attribute_names=attribute_names,
        attribute_values=attribute_values,
        degree_bins=DEGREE_BINS.copy(),
    )
