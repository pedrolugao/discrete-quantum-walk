import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from scipy.optimize import minimize

from discrete_walk import DiscreteQuantumWalk


class Fitting:
    def __init__(self, graph: list[list[int]]):
        self.graph = graph
        self.walk = DiscreteQuantumWalk(graph)
        self.node_register_dimension = 2 ** self.walk.num_qubits_nodes
        self.total_dimension = 2 ** self.walk.num_qubits_total

        self._nodes = self.walk._DiscreteQuantumWalk__nodes
        self._node_parameter_sizes = [self._nodes[node].degree ** 2 for node in range(self.walk.num_nodes)]
        self._total_parameters = int(sum(self._node_parameter_sizes))
        self._embedding = self.__buildNodeEmbedding()

    def __validateTargetUnitary(self, target_unitary: np.ndarray) -> np.ndarray:
        parsed_target = np.asarray(target_unitary, dtype=complex)

        if parsed_target.shape != (self.node_register_dimension, self.node_register_dimension):
            raise ValueError(
                "target_unitary must match node register dimension "
                f"{self.node_register_dimension}x{self.node_register_dimension}"
            )

        identity = np.eye(self.node_register_dimension, dtype=complex)

        if not np.allclose(parsed_target.conj().T @ parsed_target, identity, atol=1e-8):
            raise ValueError("target_unitary must be unitary")

        return parsed_target

    def __buildNodeEmbedding(self) -> np.ndarray:
        embedding = np.zeros((self.total_dimension, self.node_register_dimension), dtype=complex)

        for node_state in range(self.node_register_dimension):
            column = np.zeros(self.total_dimension, dtype=complex)

            binary_node = bin(node_state)[2:].zfill(self.walk.num_qubits_nodes)

            if node_state < self.walk.num_nodes and self._nodes[node_state].degree > 0:
                inverse_of_sqrt_degree = 1 / np.sqrt(self._nodes[node_state].degree)

                for connection_id in self._nodes[node_state].connection_ids:
                    binary_connection = bin(connection_id)[2:].zfill(self.walk.num_qubits_connections)
                    state = binary_node + binary_connection
                    column[int(state, 2)] += inverse_of_sqrt_degree

                norm = np.linalg.norm(column)
                if norm > 0:
                    column /= norm
            else:
                state = binary_node + ("0" * self.walk.num_qubits_connections)
                column[int(state, 2)] = 1.0

            embedding[:, node_state] = column

        return embedding

    def __flatToNodeParameters(self, flat_parameters: np.ndarray) -> dict[int, list[float]]:
        parsed = np.asarray(flat_parameters, dtype=float)

        if parsed.shape != (self._total_parameters,):
            raise ValueError(f"flat parameter vector must have {self._total_parameters} values")

        by_node = {}
        cursor = 0

        for node in range(self.walk.num_nodes):
            node_size = self._node_parameter_sizes[node]
            by_node[node] = parsed[cursor:cursor + node_size].tolist()
            cursor += node_size

        return by_node

    def __globalPhaseInvariantError(self, candidate: np.ndarray, target: np.ndarray) -> float:
        overlap = np.trace(target.conj().T @ candidate)
        phase = np.angle(overlap) if np.abs(overlap) > 0 else 0.0
        aligned_candidate = candidate * np.exp(-1j * phase)
        return float(np.linalg.norm(aligned_candidate - target, ord="fro"))

    def _buildStepUnitary(self, flat_parameters: np.ndarray) -> np.ndarray:
        parameter_by_node = self.__flatToNodeParameters(flat_parameters)
        walk = DiscreteQuantumWalk(self.graph, coin_parameters=parameter_by_node)

        step_circuit = QuantumCircuit(walk.num_qubits_total)

        for gate, application in walk.gates:
            step_circuit.append(gate, application)

        return Operator(step_circuit).data

    def effectiveNodeUnitary(self, flat_parameters: list[float] | np.ndarray, steps: int) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be a positive integer")

        parsed = np.asarray(flat_parameters, dtype=float)
        step_unitary = self._buildStepUnitary(parsed)
        repeated_steps_unitary = np.linalg.matrix_power(step_unitary, steps)

        return self._embedding.conj().T @ repeated_steps_unitary @ self._embedding

    def __fitFromInitialPoint(
        self,
        target_unitary: np.ndarray,
        steps: int,
        initial_parameters: np.ndarray,
        max_iterations: int,
        tolerance: float,
    ) -> dict[str, float | int | bool | np.ndarray]:
        def objective(flat_parameters: np.ndarray) -> float:
            effective = self.effectiveNodeUnitary(flat_parameters, steps=steps)
            return self.__globalPhaseInvariantError(effective, target_unitary)

        result = minimize(
            objective,
            x0=initial_parameters,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        best_parameters = np.asarray(result.x, dtype=float)

        return {
            "success": bool(result.success),
            "iterations": int(result.nit),
            "error": objective(best_parameters),
            "parameters": best_parameters,
        }

    def fitParameters(
        self,
        target_unitary: np.ndarray,
        steps: int,
        max_iterations: int = 300,
        tolerance: float = 1e-11,
        num_restarts: int = 8,
        random_scale: float = 0.5,
        match_tolerance: float = 1e-6,
        random_seed: int = 0,
    ) -> dict[str, float | int | bool | list[float] | dict[int, list[float]]]:
        if steps <= 0:
            raise ValueError("steps must be a positive integer")

        if num_restarts < 1:
            raise ValueError("num_restarts must be at least 1")

        parsed_target = self.__validateTargetUnitary(target_unitary)
        zero_parameters = np.zeros(self._total_parameters, dtype=float)

        initial_effective = self.effectiveNodeUnitary(zero_parameters, steps=steps)
        initial_error = self.__globalPhaseInvariantError(initial_effective, parsed_target)

        rng = np.random.default_rng(random_seed)
        starting_points = [zero_parameters]

        for _ in range(num_restarts - 1):
            starting_points.append(rng.normal(scale=random_scale, size=self._total_parameters))

        best_run = None

        for initial_parameters in starting_points:
            run = self.__fitFromInitialPoint(
                target_unitary=parsed_target,
                steps=steps,
                initial_parameters=np.asarray(initial_parameters, dtype=float),
                max_iterations=max_iterations,
                tolerance=tolerance,
            )

            if best_run is None or run["error"] < best_run["error"]:
                best_run = run

        best_parameters = np.asarray(best_run["parameters"], dtype=float)
        node_parameters = self.__flatToNodeParameters(best_parameters)
        matched = bool(best_run["error"] <= match_tolerance)

        self.walk.setCoinParametersByNode(node_parameters)

        return {
            "steps": steps,
            "success": bool(best_run["success"]),
            "matched": matched,
            "iterations": int(best_run["iterations"]),
            "initial_error": initial_error,
            "error": float(best_run["error"]),
            "flat_parameters": best_parameters.tolist(),
            "parameters_by_node": node_parameters,
        }

    def fitBestParameters(
        self,
        target_unitary: np.ndarray,
        step_candidates: list[int] | range,
        max_iterations: int = 300,
        tolerance: float = 1e-11,
        num_restarts: int = 8,
        random_scale: float = 0.5,
        match_tolerance: float = 1e-6,
        random_seed: int = 0,
    ) -> dict[str, float | int | bool | list[float] | dict[int, list[float]]]:
        parsed_target = self.__validateTargetUnitary(target_unitary)
        normalized_step_candidates = list(step_candidates)

        if normalized_step_candidates == []:
            raise ValueError("step_candidates must not be empty")

        best_result = None

        for step_index, steps in enumerate(normalized_step_candidates):
            result = self.fitParameters(
                target_unitary=parsed_target,
                steps=steps,
                max_iterations=max_iterations,
                tolerance=tolerance,
                num_restarts=num_restarts,
                random_scale=random_scale,
                match_tolerance=match_tolerance,
                random_seed=random_seed + step_index,
            )

            if best_result is None or result["error"] < best_result["error"]:
                best_result = result

        self.walk.setCoinParametersByNode(best_result["parameters_by_node"])

        return best_result


def _run_smoke_tests() -> None:
    graph = [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]

    fitter = Fitting(graph)
    dim = fitter.node_register_dimension

    identity = np.eye(dim, dtype=complex)
    cnot = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )
    swap = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )

    dt = 0.15
    adjacency = np.array(graph, dtype=float)
    continuous_walk_step = expm(-1j * dt * adjacency)

    print("Exact match sanity test")

    identity_result = fitter.fitBestParameters(
        target_unitary=identity,
        step_candidates=range(1, 7),
        max_iterations=250,
        num_restarts=8,
        random_scale=0.6,
        match_tolerance=1e-8,
    )

    print(
        f"Identity: steps={identity_result['steps']}, initial_error={identity_result['initial_error']:.6f}, "
        f"fitted_error={identity_result['error']:.6f}, matched={identity_result['matched']}"
    )

    if not identity_result["matched"]:
        raise AssertionError("Identity should be fit exactly for a suitable number of steps")

    print("\nCommon target approximations")

    approximation_tests = [
        ("CNOT", cnot),
        ("SWAP", swap),
        ("Continuous-time step", continuous_walk_step),
    ]

    for name, target in approximation_tests:
        result = fitter.fitBestParameters(
            target_unitary=target,
            step_candidates=range(1, 7),
            max_iterations=250,
            num_restarts=8,
            random_scale=0.6,
            match_tolerance=1e-6,
        )
        print(
            f"{name}: steps={result['steps']}, initial_error={result['initial_error']:.6f}, "
            f"fitted_error={result['error']:.6f}, matched={result['matched']}"
        )

        if not np.isfinite(result["error"]):
            raise AssertionError(f"{name} produced a non-finite fitting error")

        if result["error"] > result["initial_error"] + 1e-9:
            raise AssertionError(f"{name} fitting got worse than the initial point")

    one_step_cts = fitter.fitParameters(
        target_unitary=continuous_walk_step,
        steps=1,
        max_iterations=250,
        num_restarts=8,
        random_scale=0.6,
    )
    best_step_cts = fitter.fitBestParameters(
        target_unitary=continuous_walk_step,
        step_candidates=range(1, 7),
        max_iterations=250,
        num_restarts=8,
        random_scale=0.6,
    )

    print(
        f"\nContinuous-time step improvement: one_step_error={one_step_cts['error']:.6f}, "
        f"best_error={best_step_cts['error']:.6f} at steps={best_step_cts['steps']}"
    )

    if best_step_cts["error"] > one_step_cts["error"] + 1e-9:
        raise AssertionError("Step search should not perform worse than using one step")

    print("\nAll fitting smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_tests()
