import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from discrete_walk import DiscreteQuantumWalk


class Fitting:
    def __init__(self, graph: list[list[int]]):
        self.walk = DiscreteQuantumWalk(graph)

    def __validateNodeIndex(self, node: int) -> None:
        if node < 0 or node > self.walk.num_nodes - 1:
            raise ValueError(f"node must be between 0 and {self.walk.num_nodes - 1}")

    def __getNodeDegree(self, node: int) -> int:
        return self.walk._DiscreteQuantumWalk__nodes[node].degree

    def __getNodeConnectionIds(self, node: int) -> list[int]:
        return self.walk._DiscreteQuantumWalk__nodes[node].connection_ids

    def __parseParameters(self, node: int, parameters: list[float] | np.ndarray) -> np.ndarray:
        degree = self.__getNodeDegree(node)
        expected_num_parameters = degree ** 2

        parsed_parameters = np.asarray(parameters, dtype=float)

        if parsed_parameters.shape != (expected_num_parameters,):
            raise ValueError(
                f"parameters must have {expected_num_parameters} elements for node {node} (degree {degree})"
            )

        return parsed_parameters

    def __validateTargetUnitary(self, node: int, target_unitary: np.ndarray) -> np.ndarray:
        degree = self.__getNodeDegree(node)
        parsed_target = np.asarray(target_unitary, dtype=complex)

        if parsed_target.shape != (degree, degree):
            raise ValueError(
                f"target_unitary must be a {degree}x{degree} matrix for node {node} (degree {degree})"
            )

        identity = np.eye(degree, dtype=complex)

        if not np.allclose(parsed_target.conj().T @ parsed_target, identity, atol=1e-8):
            raise ValueError("target_unitary must be unitary")

        return parsed_target

    def __parametersFromHermitian(self, hermitian: np.ndarray) -> np.ndarray:
        degree = hermitian.shape[0]
        parameters = []

        for i in range(degree):
            parameters.append(float(hermitian[i][i].real))

        for i in range(degree):
            for j in range(i + 1, degree):
                parameters.append(float(hermitian[i][j].real))
                parameters.append(float(hermitian[i][j].imag))

        return np.asarray(parameters, dtype=float)

    def __localCoinUnitaryFromParameters(self, node: int, parameters: np.ndarray) -> np.ndarray:
        parsed_parameters = self.__parseParameters(node, parameters)

        full_coin_unitary = self.walk._DiscreteQuantumWalk__buildCoinUnitaryFromParameters(node, parsed_parameters)
        connection_ids = self.__getNodeConnectionIds(node)

        return full_coin_unitary[np.ix_(connection_ids, connection_ids)]

    def fitParameters(
        self,
        node: int,
        target_unitary: np.ndarray,
        steps: int,
        max_iterations: int = 300,
        tolerance: float = 1e-9,
    ) -> dict[str, int | float | bool | list[float]]:
        self.__validateNodeIndex(node)

        if steps <= 0:
            raise ValueError("steps must be a positive integer")

        degree = self.__getNodeDegree(node)
        if degree == 0:
            raise ValueError("cannot fit parameters for an isolated node")

        parsed_target = self.__validateTargetUnitary(node, target_unitary)

        principal_log = logm(parsed_target)
        hermitian_guess = (-1j / steps) * principal_log
        hermitian_guess = 0.5 * (hermitian_guess + hermitian_guess.conj().T)

        initial_parameters = self.__parametersFromHermitian(hermitian_guess)

        def objective(parameters: np.ndarray) -> float:
            local_coin = self.__localCoinUnitaryFromParameters(node, parameters)
            walk_after_steps = np.linalg.matrix_power(local_coin, steps)
            return float(np.linalg.norm(walk_after_steps - parsed_target, ord="fro"))

        initial_error = objective(initial_parameters)

        if initial_error <= tolerance:
            best_parameters = initial_parameters
            success = True
            iterations = 0
        else:
            result = minimize(
                objective,
                x0=initial_parameters,
                method="L-BFGS-B",
                options={"maxiter": max_iterations, "ftol": tolerance},
            )
            best_parameters = np.asarray(result.x, dtype=float)
            success = bool(result.success)
            iterations = int(result.nit)

        final_error = objective(best_parameters)

        self.walk.setCoinParameters(node, best_parameters.tolist())

        return {
            "node": node,
            "steps": steps,
            "success": success,
            "iterations": iterations,
            "error": final_error,
            "parameters": best_parameters.tolist(),
        }

    def fitParametersByNode(
        self,
        target_unitary_by_node: dict[int, np.ndarray],
        steps: int,
        max_iterations: int = 300,
        tolerance: float = 1e-9,
    ) -> dict[int, dict[str, int | float | bool | list[float]]]:
        fit_results = {}

        for node, target_unitary in target_unitary_by_node.items():
            fit_results[node] = self.fitParameters(
                node=node,
                target_unitary=target_unitary,
                steps=steps,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )

        return fit_results

    def localCoinUnitary(self, node: int, parameters: list[float] | np.ndarray) -> np.ndarray:
        self.__validateNodeIndex(node)
        return self.__localCoinUnitaryFromParameters(node, np.asarray(parameters, dtype=float))


def _run_smoke_tests() -> None:
    graph = [
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
    ]

    fitter = Fitting(graph)

    identity = np.eye(2, dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

    dt = 0.2
    hamiltonian = np.array([[0, 1], [1, 0]], dtype=complex)
    continuous_walk_step = expm(-1j * dt * hamiltonian)

    test_cases = [
        ("Identity", identity, 3),
        ("Pauli-X", pauli_x, 1),
        ("Hadamard", hadamard, 1),
        ("Continuous-time step", continuous_walk_step, 4),
    ]

    for name, target, steps in test_cases:
        fitter.walk.clearCoinParameters()
        result = fitter.fitParameters(node=0, target_unitary=target, steps=steps, tolerance=1e-10)
        fitted_local_coin = fitter.localCoinUnitary(node=0, parameters=result["parameters"])
        reconstructed_target = np.linalg.matrix_power(fitted_local_coin, steps)
        error = float(np.linalg.norm(reconstructed_target - target, ord="fro"))

        print(f"{name}: error={error:.3e}, success={result['success']}, iterations={result['iterations']}")

        if error > 1e-6:
            raise AssertionError(f"{name} fit error is too high: {error}")

    by_node_results = fitter.fitParametersByNode(
        target_unitary_by_node={0: pauli_x, 2: hadamard},
        steps=1,
        tolerance=1e-10,
    )

    for node, target in ((0, pauli_x), (2, hadamard)):
        fitted_local_coin = fitter.localCoinUnitary(node=node, parameters=by_node_results[node]["parameters"])
        error = float(np.linalg.norm(fitted_local_coin - target, ord="fro"))
        print(f"Node {node} independent fit: error={error:.3e}")
        if error > 1e-6:
            raise AssertionError(f"Node {node} fit error is too high: {error}")

    print("All fitting smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_tests()
