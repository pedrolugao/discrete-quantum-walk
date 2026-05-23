# QGEL - Quantum Graph Extended Library

QGEL is a general-purpose quantum graph library. It offers tools to visualize, interpret, simulate and work with graphs using quantum technology.

Built upon Qiskit, Matplotlib and NetworkX, QGEL aims to be a single-point and simple (as much as quantum computing can be) extensive library, meaning that all work envolving graphs can be done using only the functions provided by it, without the need for external tools.

## How does it work?

## What problems does it solve?

## Examples

### Parametrized coin

`DiscreteQuantumWalk` now accepts optional per-node coin parameters:

```python
# node 0 has degree 3 in this graph
node_0_degree = int(sum(graph[0]))

coin_parameters = {
    0: [0.0] * (node_0_degree ** 2),
}

walk = DiscreteQuantumWalk(graph, coin_parameters=coin_parameters)
```

If a node has parameters, its coin is a parametrized unitary `d x d` matrix, where `d` is that node degree. The matrix is embedded in the connection register and acts only on that node's neighbor/connection subspace.  
Nodes without explicit parameters keep the default Grover diffusion coin behavior.
