import gzip
import io
import json
import os
import shutil
import ssl
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlopen, urlretrieve

import networkx as nx


def _create_unverified_context():
    """Create SSL context that doesn't verify certificates (for HPC clusters with proxy issues)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


@dataclass
class DatasetInfo:
    name: str
    description: str
    url: str
    filename: str
    num_nodes_approx: int
    num_edges_approx: int
    format: str
    compressed: bool = True
    citation: Optional[str] = None
    license: Optional[str] = None
    preprocessing: Optional[Callable[[Path], Path]] = None


DATA_DIR = Path(os.environ.get("GRAPHCUBES_DATA_DIR", Path.home() / ".cache" / "graphcubes"))


def _preprocess_bron(raw_path: Path) -> Path:
    output_path = raw_path.parent / "bron.json"
    if output_path.exists():
        return output_path

    with zipfile.ZipFile(raw_path) as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        if not json_files:
            raise ValueError("No JSON file found in BRON archive")

        bron_file = None
        for f in json_files:
            if "bron" in f.lower() or "graph" in f.lower():
                bron_file = f
                break
        if not bron_file:
            bron_file = json_files[0]

        with zf.open(bron_file) as src:
            data = json.load(src)

        with open(output_path, "w") as f:
            json.dump(data, f)

    return output_path


def _preprocess_snap_edgelist(raw_path: Path) -> Path:
    output_path = raw_path.parent / (raw_path.stem.replace(".txt", "") + ".edges")
    if output_path.exists():
        return output_path

    if str(raw_path).endswith(".gz"):
        with gzip.open(raw_path, "rt") as src:
            with open(output_path, "w") as dst:
                for line in src:
                    if not line.startswith("#"):
                        dst.write(line)
    else:
        shutil.copy(raw_path, output_path)

    return output_path


def _preprocess_ogb(raw_path: Path) -> Path:
    import numpy as np

    output_path = raw_path.parent / "graph.edges"
    if output_path.exists():
        return output_path

    extract_dir = raw_path.parent / "extracted"
    with zipfile.ZipFile(raw_path) as zf:
        zf.extractall(extract_dir)

    edge_file = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if "edge" in f.lower() and f.endswith(".csv"):
                edge_file = Path(root) / f
                break
            elif f == "edge.csv.gz":
                edge_file = Path(root) / f
                break

    if edge_file is None:
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if "edge_index" in f and f.endswith(".npy"):
                    edge_file = Path(root) / f
                    break

    if edge_file is None:
        raise ValueError(f"Could not find edge file in {extract_dir}")

    with open(output_path, "w") as out:
        if str(edge_file).endswith(".npy"):
            edges = np.load(edge_file)
            for i in range(edges.shape[1]):
                out.write(f"{edges[0, i]}\t{edges[1, i]}\n")
        elif str(edge_file).endswith(".gz"):
            with gzip.open(edge_file, "rt") as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        out.write(f"{parts[0]}\t{parts[1]}\n")
        else:
            with open(edge_file) as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        out.write(f"{parts[0]}\t{parts[1]}\n")

    return output_path


DATASETS: dict[str, DatasetInfo] = {
    "bron": DatasetInfo(
        name="bron",
        description="BRON: Linked Threat Intelligence graph connecting CVEs, CWEs, CAPEC, and MITRE ATT&CK",
        url="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/s2sw4ck42n-1.zip",
        filename="bron_raw.zip",
        num_nodes_approx=30000,
        num_edges_approx=150000,
        format="json",
        citation="Hemberg et al., Linking Threat Tactics, Techniques, and Patterns with Defensive Weaknesses, 2020",
        license="CC BY 4.0",
        preprocessing=_preprocess_bron,
    ),
    "twitch": DatasetInfo(
        name="twitch",
        description="Twitch social network (English users)",
        url="https://snap.stanford.edu/data/twitch_gamers.zip",
        filename="twitch_raw.zip",
        num_nodes_approx=168000,
        num_edges_approx=6800000,
        format="edgelist",
        citation="Rozemberczki et al., Multi-scale Attributed Node Embedding, 2019",
        license="Public domain",
    ),
    "github": DatasetInfo(
        name="github",
        description="GitHub developer social network",
        url="https://snap.stanford.edu/data/git_web_ml.zip",
        filename="github_raw.zip",
        num_nodes_approx=37700,
        num_edges_approx=578000,
        format="edgelist",
        citation="Rozemberczki et al., Multi-scale Attributed Node Embedding, 2019",
        license="Public domain",
    ),
    "deezer": DatasetInfo(
        name="deezer",
        description="Deezer Europe social network",
        url="https://snap.stanford.edu/data/deezer_europe.zip",
        filename="deezer_raw.zip",
        num_nodes_approx=28000,
        num_edges_approx=185000,
        format="edgelist",
        citation="Rozemberczki et al., Gemsec, 2019",
        license="Public domain",
    ),
    "facebook": DatasetInfo(
        name="facebook",
        description="Facebook page-page network",
        url="https://snap.stanford.edu/data/facebook_large.zip",
        filename="facebook_raw.zip",
        num_nodes_approx=22000,
        num_edges_approx=340000,
        format="edgelist",
        citation="Rozemberczki et al., Multi-scale Attributed Node Embedding, 2019",
        license="Public domain",
    ),
    "wiki": DatasetInfo(
        name="wiki",
        description="Wikipedia hyperlink network (English)",
        url="https://snap.stanford.edu/data/wiki-topcats.txt.gz",
        filename="wiki_raw.txt.gz",
        num_nodes_approx=1800000,
        num_edges_approx=28500000,
        format="edgelist",
        compressed=True,
        citation="Yin et al., Local Higher-order Graph Clustering, 2017",
        license="Public domain",
        preprocessing=_preprocess_snap_edgelist,
    ),
    "reddit": DatasetInfo(
        name="reddit",
        description="Reddit hyperlink network between subreddits",
        url="https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv",
        filename="reddit_raw.tsv",
        num_nodes_approx=55000,
        num_edges_approx=860000,
        format="edgelist",
        compressed=False,
        citation="Kumar et al., Community Interaction and Conflict on the Web, 2018",
        license="Public domain",
    ),
    "amazon": DatasetInfo(
        name="amazon",
        description="Amazon product co-purchasing network",
        url="https://snap.stanford.edu/data/amazon0601.txt.gz",
        filename="amazon_raw.txt.gz",
        num_nodes_approx=403000,
        num_edges_approx=3390000,
        format="edgelist",
        compressed=True,
        citation="Leskovec et al., Graphs over Time, 2005",
        license="Public domain",
        preprocessing=_preprocess_snap_edgelist,
    ),
    "youtube": DatasetInfo(
        name="youtube",
        description="YouTube social network",
        url="https://snap.stanford.edu/data/com-youtube.ungraph.txt.gz",
        filename="youtube_raw.txt.gz",
        num_nodes_approx=1130000,
        num_edges_approx=3000000,
        format="edgelist",
        compressed=True,
        citation="Yang and Leskovec, Defining and Evaluating Network Communities, 2012",
        license="Public domain",
        preprocessing=_preprocess_snap_edgelist,
    ),
    "dblp": DatasetInfo(
        name="dblp",
        description="DBLP collaboration network",
        url="https://snap.stanford.edu/data/com-dblp.ungraph.txt.gz",
        filename="dblp_raw.txt.gz",
        num_nodes_approx=317000,
        num_edges_approx=1050000,
        format="edgelist",
        compressed=True,
        citation="Yang and Leskovec, Defining and Evaluating Network Communities, 2012",
        license="Public domain",
        preprocessing=_preprocess_snap_edgelist,
    ),
}


def list_datasets() -> list[dict]:
    return [
        {
            "name": info.name,
            "description": info.description,
            "nodes": f"~{info.num_nodes_approx:,}",
            "edges": f"~{info.num_edges_approx:,}",
            "license": info.license,
        }
        for info in DATASETS.values()
    ]


def get_dataset_path(name: str, data_dir: Optional[Path] = None) -> Path:
    if data_dir is None:
        data_dir = DATA_DIR
    return data_dir / name


def _download_with_progress(url: str, path: Path) -> None:
    print(f"Downloading {url}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Try with SSL verification first, fall back to unverified for HPC clusters
    try:
        _download_with_ssl(url, path, verify_ssl=True)
    except ssl.SSLCertVerificationError:
        print("  SSL verification failed, retrying without verification (common on HPC clusters)...")
        _download_with_ssl(url, path, verify_ssl=False)
    print()


def _download_with_ssl(url: str, path: Path, verify_ssl: bool = True) -> None:
    """Download with optional SSL verification."""
    if verify_ssl:
        context = None
    else:
        context = _create_unverified_context()

    with urlopen(url, context=context) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 8192
        downloaded = 0

        with open(path, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = downloaded * 100 / total_size
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  {pct:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)


def _preprocess_zip_edgelist(raw_path: Path, dataset_name: str) -> Path:
    output_path = raw_path.parent / "graph.edges"
    if output_path.exists():
        return output_path

    extract_dir = raw_path.parent / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(raw_path) as zf:
        zf.extractall(extract_dir)

    edge_file = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            f_lower = f.lower()
            if any(x in f_lower for x in ["edge", "link", "graph"]) and \
               any(f_lower.endswith(ext) for ext in [".csv", ".txt", ".edges", ".tsv"]):
                edge_file = Path(root) / f
                break
        if edge_file:
            break

    if edge_file is None:
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith((".csv", ".txt", ".edges", ".tsv")) and not f.startswith("."):
                    edge_file = Path(root) / f
                    break
            if edge_file:
                break

    if edge_file is None:
        raise ValueError(f"Could not find edge file in {extract_dir}")

    with open(output_path, "w") as out:
        with open(edge_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()
                if len(parts) >= 2:
                    if parts[0].lower() in ("source", "src", "from", "node1", "id"):
                        continue
                    out.write(f"{parts[0]}\t{parts[1]}\n")

    return output_path


def download_dataset(
    name: str,
    data_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    info = DATASETS[name]
    if data_dir is None:
        data_dir = DATA_DIR

    dataset_dir = data_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_path = dataset_dir / info.filename
    processed_path = dataset_dir / "graph.edges"

    if processed_path.exists() and not force:
        print(f"Dataset {name} already downloaded: {processed_path}")
        return processed_path

    json_path = dataset_dir / "graph.json"
    if json_path.exists() and not force:
        print(f"Dataset {name} already downloaded: {json_path}")
        return json_path

    if not raw_path.exists() or force:
        _download_with_progress(info.url, raw_path)
    else:
        print(f"Raw file exists: {raw_path}")

    print("Processing...")
    if info.preprocessing:
        final_path = info.preprocessing(raw_path)
    elif info.filename.endswith(".zip"):
        final_path = _preprocess_zip_edgelist(raw_path, name)
    elif info.filename.endswith(".gz"):
        final_path = _preprocess_snap_edgelist(raw_path)
    else:
        final_path = dataset_dir / "graph.edges"
        if not final_path.exists():
            with open(raw_path) as src:
                with open(final_path, "w") as dst:
                    for line in src:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "\t" in line:
                            parts = line.split("\t")
                        else:
                            parts = line.split()
                        if len(parts) >= 2:
                            dst.write(f"{parts[0]}\t{parts[1]}\n")

    print(f"Dataset ready: {final_path}")
    return final_path


def load_dataset(
    name: str,
    data_dir: Optional[Path] = None,
    download: bool = True,
) -> tuple[nx.Graph, dict]:
    from training.graph_loader import load_graph, GraphMetadata

    if data_dir is None:
        data_dir = DATA_DIR

    dataset_dir = data_dir / name

    for filename in ["graph.edges", "graph.json", "bron.json"]:
        path = dataset_dir / filename
        if path.exists():
            G, metadata = load_graph(path, name=name)
            return G, metadata.to_dict()

    if download:
        path = download_dataset(name, data_dir)
        G, metadata = load_graph(path, name=name)
        return G, metadata.to_dict()

    raise FileNotFoundError(f"Dataset {name} not found. Run download_dataset('{name}') first.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage graph datasets")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available datasets")

    dl_parser = subparsers.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("name", help="Dataset name")
    dl_parser.add_argument("--force", action="store_true", help="Re-download if exists")
    dl_parser.add_argument("--data-dir", type=Path, help="Data directory")

    info_parser = subparsers.add_parser("info", help="Show dataset info")
    info_parser.add_argument("name", help="Dataset name")

    args = parser.parse_args()

    if args.command == "list":
        print("\nAvailable datasets:")
        print("-" * 80)
        for ds in list_datasets():
            print(f"  {ds['name']:12} | {ds['nodes']:>12} nodes | {ds['edges']:>15} edges")
            print(f"               | {ds['description'][:60]}")
            print()

    elif args.command == "download":
        download_dataset(args.name, data_dir=args.data_dir, force=args.force)

    elif args.command == "info":
        if args.name not in DATASETS:
            print(f"Unknown dataset: {args.name}")
            return
        info = DATASETS[args.name]
        print(f"\nDataset: {info.name}")
        print(f"Description: {info.description}")
        print(f"Approximate size: {info.num_nodes_approx:,} nodes, {info.num_edges_approx:,} edges")
        print(f"Format: {info.format}")
        print(f"License: {info.license}")
        if info.citation:
            print(f"Citation: {info.citation}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
