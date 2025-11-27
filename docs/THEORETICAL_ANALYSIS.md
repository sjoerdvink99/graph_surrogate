# Theoretical Analysis: GraphSurrogate

This document provides theoretical foundations for GraphSurrogate, including approximation guarantees, sample complexity bounds, and connections to graph neural network expressiveness.

## 1. Problem Formalization

### 1.1 Query Space Definition

Let $\mathcal{G} = (V, E, \mathcal{A})$ be an attributed graph with:

- $V$: set of $n$ nodes
- $E \subseteq V \times V$: set of $m$ edges
- $\mathcal{A}: V \to \mathcal{T} \times \mathcal{F}$: node attribute function mapping to type $\mathcal{T}$ and features $\mathcal{F}$

We define two query families:

**Count Queries** $Q_C$: For node $v \in V$, radius $r \in \mathbb{Z}^+$, and optional filter $\phi$:
$$q_C(v, r, \phi) = |\{u \in N_r(v) : \phi(u) = \text{true}\}|$$
where $N_r(v)$ is the $r$-hop neighborhood of $v$.

**Distance Queries** $Q_D$: For node $v \in V$, target type $t \in \mathcal{T}$, and max hops $h$:
$$q_D(v, t, h) = \min_{u: \mathcal{A}(u)_\mathcal{T} = t} d(v, u) \text{ s.t. } d(v,u) \leq h$$
where $d(v,u)$ is the shortest path distance.

### 1.2 Query Encoding

We encode queries as fixed-dimensional vectors $\mathbf{x} \in \mathbb{R}^d$ using one-hot encodings:
$$\mathbf{x} = [\text{type}(v); \text{deg\_bin}(v); r; \phi; t; h; \text{query\_type}]$$

The encoding dimension is:
$$d = |\mathcal{T}| + |\mathcal{B}| + |\mathcal{R}| + |\mathcal{F}| + |\mathcal{T}| + |\mathcal{H}| + 2$$
where $\mathcal{B}$ are degree bins and $\mathcal{R}, \mathcal{H}$ are radius/hop options.

## 2. Approximation Theory

### 2.1 Universal Approximation for Graph Queries

**Theorem 1 (Existence of Approximator)**: For any graph $\mathcal{G}$ with bounded degree $\Delta$, and any $\epsilon > 0$, there exists an MLP $f_\theta: \mathbb{R}^d \to \mathbb{R}$ with $O(\text{poly}(n, \Delta, 1/\epsilon))$ parameters such that for all valid queries $q$:
$$|f_\theta(\text{encode}(q)) - q(\mathcal{G})| < \epsilon$$

_Proof sketch_: The query space $\mathcal{Q}$ is finite (bounded by combinations of discrete encoding components). By the universal approximation theorem, MLPs can approximate any continuous function on a compact domain. Since we're approximating a finite function table, an MLP with sufficient capacity can memorize all query-result pairs with arbitrary precision.

### 2.2 Effective Dimension of Query Space

**Lemma 1**: The effective dimensionality of the query-result mapping is bounded by the graph's structural complexity.

For count queries with radius $r$, the result depends on:

1. The $r$-hop ego network structure around the start node
2. The attribute distribution in this neighborhood

**Definition (Structural Equivalence)**: Two nodes $u, v$ are $(r, \phi)$-equivalent if their $r$-hop neighborhoods have identical structure and attribute distributions.

**Theorem 2**: If $\mathcal{G}$ has $k$ equivalence classes under $(r, \phi)$-equivalence, then an MLP with $O(k \cdot d)$ parameters suffices for exact count query prediction.

_Implications_: Real-world graphs exhibit regularity (similar neighborhoods for nodes of similar degree/type), making $k \ll n$ and enabling efficient learning.

## 3. Sample Complexity Bounds

### 3.1 PAC Learning Framework

We analyze sample complexity in the PAC (Probably Approximately Correct) framework.

**Theorem 3 (Sample Complexity for Count Queries)**: To learn count query predictions with error $\epsilon$ and confidence $1-\delta$ on a graph with maximum neighborhood size $M$, it suffices to sample:
$$N = O\left(\frac{M^2}{\epsilon^2} \log\frac{|\mathcal{Q}|}{\delta}\right)$$
queries, where $|\mathcal{Q}|$ is the query space size.

_Proof_:

1. Count queries have bounded range $[0, M]$ where $M \leq \Delta^r$ for radius $r$ and max degree $\Delta$
2. By Hoeffding's inequality, each query-type pair requires $O(M^2/\epsilon^2)$ samples for concentration
3. Union bound over $|\mathcal{Q}|$ query configurations gives the result

### 3.2 Practical Sample Complexity

For typical graph parameters:

- $|\mathcal{T}| = 5$ node types
- $|\mathcal{B}| = 5$ degree bins
- $|\mathcal{R}| = 3$ radii
- Binary filter (on/off)

The query space is $|\mathcal{Q}| = O(|\mathcal{T}| \cdot |\mathcal{B}| \cdot |\mathcal{R}| \cdot 2) = O(150)$ for count queries.

**Corollary**: With $N = 100,000$ training queries, we have approximately $667$ samples per query configuration, sufficient for convergence under typical graph statistics.

### 3.3 Generalization Bound

**Theorem 4 (Rademacher Complexity Bound)**: For an MLP with $L$ layers, width $w$, and spectral norm bound $\rho$ per layer, the generalization gap is bounded by:
$$\mathcal{R}_N(f) \leq \frac{2\rho^L \sqrt{d}}{\sqrt{N}}$$

For our architecture ($L=3$, $w=128$, $\rho \approx 1$ with normalization):
$$\text{Gen. Gap} \leq O\left(\frac{\sqrt{d}}{\sqrt{N}}\right) \approx 0.02 \text{ for } N=100K, d=50$$

## 4. Connection to GNN Expressiveness

### 4.1 Relationship to 1-WL Test

Graph Neural Networks are bounded by the 1-dimensional Weisfeiler-Leman (1-WL) graph isomorphism test. Our approach differs fundamentally:

**Key Insight**: GraphSurrogate does not learn node embeddings or graph structure directly. Instead, it learns a mapping from _query descriptions_ to _query results_, bypassing GNN expressiveness limitations.

**Theorem 5**: GraphSurrogate can distinguish query results that 1-WL equivalent nodes would produce identically, provided the queries involve discriminating features.

_Example_: Two nodes with identical 1-WL colors but different 3-hop neighborhood sizes produce different count query results. GraphSurrogate learns this difference through the (node_type, degree_bin) encoding.

### 4.2 Computational Separation

**Proposition**: For queries computable in $O(n^k)$ time on the graph, GraphSurrogate provides $O(1)$ inference time after $O(n^k \cdot S)$ preprocessing (query sampling) and $O(S)$ training time.

This represents a favorable time-space tradeoff when:

1. Many queries are needed (amortized cost decreases)
2. Graph is static or slowly changing
3. Query latency is critical

## 5. Limitations and Failure Modes

### 5.1 Theoretical Limitations

**Limitation 1 (Information Bottleneck)**: The encoding discards specific node identity, relying on (type, degree_bin) as a proxy. This causes errors when:

- Two nodes of same type/degree have very different neighborhoods
- The degree binning is too coarse for the graph's degree distribution

**Limitation 2 (Distribution Shift)**: The learned model is specific to the training graph's structure. For a graph $\mathcal{G}'$ with different structure:
$$\mathbb{E}_{q \sim \mathcal{Q}}[|f_\theta(q) - q(\mathcal{G}')|] \propto D_{KL}(\mathcal{G} \| \mathcal{G}')$$
where $D_{KL}$ measures structural divergence.

**Limitation 3 (Rare Query Patterns)**: For query configurations with few training examples, variance is high:
$$\text{Var}[f_\theta(q)] \propto \frac{1}{N_q}$$
where $N_q$ is the number of training samples for query type $q$.

### 5.2 When the Approach Fails

1. **Adversarial queries**: Hand-crafted queries targeting model weaknesses
2. **Highly irregular graphs**: Graphs where (type, degree) doesn't predict neighborhood structure
3. **Dynamic graphs**: Rapidly changing graphs invalidate learned statistics
4. **Out-of-distribution queries**: Query parameters not seen during training

## 6. Comparison with Learned Cardinality Estimation

### 6.1 Relationship to Database Methods

Learned cardinality estimation methods (MSCN, NeuroCard, DeepDB) solve a related but distinct problem:

| Aspect    | Cardinality Estimation     | GraphSurrogate                   |
| --------- | -------------------------- | -------------------------------- |
| Input     | SQL query (joins, filters) | Graph query (neighborhood, path) |
| Structure | Relational tables          | Graph topology                   |
| Output    | Tuple count estimate       | Aggregate over neighborhoods     |
| Challenge | Join selectivity           | Exponential neighborhood growth  |

### 6.2 Shared Techniques

Both domains benefit from:

- Learned query embeddings
- Multi-task learning (different query types)
- Sample-based training
- Uncertainty estimation for query optimization

### 6.3 Key Differences

GraphSurrogate handles:

- **Recursive structure**: k-hop neighborhoods (exponential growth)
- **Path queries**: Shortest path computations
- **Local context**: Start node properties affect results

These require graph-specific encoding not present in SQL cardinality estimators.

## 7. Implications for Query Optimization

### 7.1 Accuracy Requirements

For query optimization, relative ordering matters more than absolute accuracy:
$$P(\text{rank}(f_\theta(q_1)) = \text{rank}(q_1(\mathcal{G}))) > 1 - \delta$$

**Theorem 6**: If $|f_\theta(q) - q(\mathcal{G})| < \epsilon$ for all queries and $|q_1(\mathcal{G}) - q_2(\mathcal{G})| > 2\epsilon$, then ranking is preserved.

### 7.2 Integration Strategy

GraphSurrogate predictions can inform:

1. **Index selection**: Prioritize indexes for high-count neighborhoods
2. **Join ordering**: Use predicted sizes for cost estimation
3. **Parallel execution**: Allocate resources based on predicted complexity

## 8. Future Theoretical Directions

1. **Tighter sample complexity**: Instance-dependent bounds using graph properties
2. **Online learning**: Bounds for incremental model updates
3. **Compositional queries**: Handling query combinations
4. **Provable robustness**: Certified bounds on adversarial perturbations

## References

1. Weisfeiler, B., Leman, A. (1968). A reduction of a graph to a canonical form.
2. Xu, K., et al. (2019). How Powerful are Graph Neural Networks?
3. Kipf, T., Welling, M. (2017). Semi-Supervised Classification with GCNs.
4. Duvenaud, D., et al. (2015). Convolutional Networks on Graphs for Learning Molecular Fingerprints.
5. Hilprecht, B., et al. (2020). DeepDB: Learn from Data, not from Queries!
6. Yang, Z., et al. (2020). NeuroCard: One Cardinality Estimator for All Tables.
7. Kipf, A., et al. (2019). Learned Cardinalities: Estimating Correlated Joins with Deep Learning.
