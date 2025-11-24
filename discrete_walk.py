from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.standard_gates import XGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


import networkx as nx

from qiskit.quantum_info import Statevector

class DiscreteTimeWalk:

    def __init__(self, graph : list[list[int]] | nx.Graph):

        if type(graph) == nx.Graph:
            self.networkx_graph = graph
            graph = nx.adjacency_matrix(graph).toarray()

        else:
            self.networkx_graph = nx.from_numpy_array(np.array(graph))

        self.adj_matrix = graph
        self.num_nodes = len(self.adj_matrix)
        self.__nodes = {}
        self.__connections : dict[str, int] = {}
        self.num_connections = 0

        for i in range(self.num_nodes):
            self.__nodes[i] = {"degree": 0, "connections": []}

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                if (self.adj_matrix[i][j]):
                    self.__connections[self.num_connections] = {"node1": i, "node2": j}
                    self.__nodes[i]["degree"] += 1
                    self.__nodes[i]["connections"].append(self.num_connections)
                    self.__nodes[j]["degree"] += 1
                    self.__nodes[j]["connections"].append(self.num_connections)
                    self.num_connections += 1

        self.gates : list[list[int | list[int]]] = []

        self.log_num_nodes = int(np.ceil(np.log2(self.num_nodes)))

        self.log_num_connections = int(np.ceil(np.log2(self.num_connections)))

        self.total_num_qubits = self.log_num_nodes + self.log_num_connections

        self.__CXGate = XGate().control(self.log_num_connections-1)
        self.__application_list_CXGate = list(range(self.log_num_connections))

        self.__application_list_coin = list(range(self.log_num_connections, self.total_num_qubits)) + list(range(self.log_num_connections))

        self.__application_list_shift = list(range(self.total_num_qubits))

        qr_nodes = QuantumRegister(self.log_num_nodes, 'q')
        qr_connections = QuantumRegister(self.log_num_connections, 'l')
        cr = ClassicalRegister(self.log_num_nodes, 'c')

        self.probabilities : list[list[float]] = []
        self.steps = 0
        self.qc = QuantumCircuit(qr_connections, qr_nodes, cr)



    def __prepareCircuit(self) -> None:

        amplitudes = []

        num_possibilities = 2 ** (self.total_num_qubits)

        for i in range(num_possibilities):
            amplitudes.append(0)

        for node in range(self.num_nodes):

            inverse_of_sqrt_degree = 1/np.sqrt(self.__nodes[node]["degree"])

            for connection in self.__nodes[node]["connections"]:
                binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                binary_connection = bin(connection)[2:].zfill(self.log_num_connections)
                state = binary_node + binary_connection
                amplitudes[int(state, 2)] = inverse_of_sqrt_degree

        inverse_of_sqrt_num_nodes = 1/np.sqrt(self.num_nodes)

        for i in range(num_possibilities):
            if amplitudes[i] != 0: # mathematically doesn't make a difference but is prittier to print and skip some divisions and multiplications
                amplitudes[i] *= inverse_of_sqrt_num_nodes

        self.qc.prepare_state(amplitudes)


    # Basically the same function but without a for loop
    def __prepareCircuitFromStartingNode(self, node : int) -> None:

        amplitudes = []

        num_possibilities = 2 ** (self.total_num_qubits)

        for i in range(num_possibilities):
            amplitudes.append(0)
        
        inverse_of_sqrt_degree = 1/np.sqrt(self.__nodes[node]["degree"])

        for connection in self.__nodes[node]["connections"]:
                binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                binary_connection = bin(connection)[2:].zfill(self.log_num_connections)
                state = binary_node + binary_connection
                amplitudes[int(state, 2)] = inverse_of_sqrt_degree

        self.qc.prepare_state(amplitudes)



    def __getAmplitudeOfNode(self, node : int) -> list[float]:

        amplitudes = []

        inverse_of_sqrt_degree = 1/np.sqrt(self.__nodes[node]["degree"])

        for i in range(2 ** self.log_num_connections):
            amplitudes.append(0)

        for connection in self.__nodes[node]["connections"]:
            amplitudes[connection] = inverse_of_sqrt_degree

        return amplitudes



    def __addCoin(self, node : int) -> None:

        binary_node = bin(node)[2:].zfill(self.log_num_nodes)

        sub_qc_coin = QuantumCircuit(self.log_num_connections)

        amplitudes = self.__getAmplitudeOfNode(node)

        # Apply Si dagger
        sub_qc_coin.prepare_state(amplitudes).inverse()

        for i in range(0, self.log_num_connections):
            sub_qc_coin.x(i)

        sub_qc_coin.h(0)
        sub_qc_coin.append(self.__CXGate, self.__application_list_CXGate[::-1])
        sub_qc_coin.h(0)

        for i in range(0, self.log_num_connections):
            sub_qc_coin.x(i)

        # Apply Si
        sub_qc_coin.prepare_state(amplitudes)

        coin_gate = sub_qc_coin.to_gate().control(self.log_num_nodes, label=f"C{node}", ctrl_state=binary_node)

        self.gates.append([coin_gate, self.__application_list_coin])



    def __addShift(self, connection : int) -> None:

        binary_connection = bin(connection)[2:].zfill(self.log_num_connections)

        sub_qc_shift = QuantumCircuit(self.log_num_nodes)

        binary_node1 = bin(self.__connections[connection]["node1"])[2:].zfill(self.log_num_nodes)[::-1] #getting little endian representation
        binary_node2 = bin(self.__connections[connection]["node2"])[2:].zfill(self.log_num_nodes)[::-1] #getting little endian representation

        for qubit in range(self.log_num_nodes):
            if binary_node1[qubit] != binary_node2[qubit]:
                sub_qc_shift.x(qubit)

        shift_gate = sub_qc_shift.to_gate().control(self.log_num_connections, label=f"S{connection}", ctrl_state=binary_connection)

        self.gates.append([shift_gate, self.__application_list_shift])


    # TODO find a way to optimize this, THIS IS *****SLOW*****
    def __getProbabilities(self) -> None:

        sv = Statevector(self.qc)
        prob_dict = sv.probabilities_dict()

        probability_list : list[float] = []

        for node in range(self.num_nodes):

            prob = 0

            for connection in range(self.num_connections):

                binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                binary_connection = bin(connection)[2:].zfill(self.log_num_connections)

                full_binary = binary_node + binary_connection

                prob += prob_dict[full_binary] if full_binary in prob_dict else 0

            probability_list.append(prob)

        self.probabilities.append(probability_list)

        """qc = self.qc.copy()
        qc.measure_all()

        shots = 1024#16384

        sim_statevector = AerSimulator(method='statevector')
        job_statevector = sim_statevector.run(transpile(qc, sim_statevector), shots=shots)
        counts_statevector = job_statevector.result().get_counts()

        probability_list : list[float] = []
        probability_list = [0.0 for _ in range(self.num_nodes)]

        for binary in counts_statevector:
            node_bin = binary[:self.log_num_nodes]
            probability_list[int(node_bin, 2)] += counts_statevector[binary] / shots

        self.probabilities.append(probability_list)"""



    def simulate(self, steps : int, register_all_probabilities : bool, starting_node : int = -1, state_prep_list : list[float] = []) -> None:

        if steps == 0:
            raise ValueError("steps must be a positive integer")

        if state_prep_list != [] and starting_node != -1:
            raise ValueError("starting_node and state_prep_list are mutually exclusive, please choose one")

        # Decided to simply ignore negative numbers, might change it in the future
        elif starting_node >= 0:
            if starting_node > self.num_nodes-1:
                raise ValueError(f"starting_node must be between 0 and {self.num_nodes-1}")
            self.__prepareCircuitFromStartingNode(starting_node)

        elif state_prep_list != []:
            if len(state_prep_list) != 2 ** (self.total_num_qubits):
                raise ValueError(f"state_prep_list must have {2 ** (self.total_num_qubits)} elements, one for each possible binary state")
            self.qc.prepare_state(state_prep_list)

        else:
            self.__prepareCircuit()

        self.steps = steps

        for node in range(self.num_nodes):
            self.__addCoin(node)

        for connection in range(self.num_connections):
            self.__addShift(connection)

        for _ in range(0, steps):
            for gate in range(len(self.gates)):
                self.qc.append(self.gates[gate][0], self.gates[gate][1]) # Appending the gate (gates[gate][0]) to it's application list (gates[gate][1])

            if register_all_probabilities:
                self.__getProbabilities()

        if not register_all_probabilities:
            self.__getProbabilities()


    # TODO(?) make this prettier
    def plotProbabilities(self, print_connection_mapping : bool) -> None:

        if self.probabilities == []:
            raise ValueError("No probabilities available")

        if len(self.probabilities) == 1: # In case of only registering the final probability
            plt.bar(range(self.num_nodes), self.probabilities[0])

        else:
            x = []
            y = []
            z = [0 for i in range(self.num_nodes * self.steps)]

            dx = [0.5 for i in range(self.num_nodes * self.steps)]
            dy = [0.5 for i in range(self.num_nodes * self.steps)]
            dz = []

            fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
            for i in range(self.steps):
                x += [j - 0.25 for j in range(self.num_nodes)]
                y += [i + 0.75 for k in range(self.num_nodes)]
                dz += self.probabilities[i]

            ax.set_xlabel("nodes")
            ax.set_xticks(list(range(self.num_nodes)))

            ax.set_ylabel("steps")
            ax.set_yticks(list(range(1, self.steps + 1)))

            ax.set_zlabel("probability")

            cmap = cm.inferno
            norm = Normalize(vmin=min(dz), vmax=max(dz))
            colors = cmap(norm(dz))

            ax.bar3d(x, y, z, dx, dy, dz, shade=True, color=colors, zsort='max')

        if print_connection_mapping:
            print("\n\t|NODE_1> <---[CONNECTION_ID]---> |NODE_2>\n")
            for i in range(self.num_connections):
                print(f"|{self.__connections[i]["node1"]}> <---[{i} / {bin(i)[2:].zfill(self.log_num_connections)}]---> |{self.__connections[i]["node2"]}>")

        plt.show()


    # Really just a wrapper for the nx draw function, but it's good for simplicity and to maintain the lib self contained
    def draw(self, show_labels : bool = True) -> None:

        nx.draw(self.networkx_graph, with_labels=show_labels)
        plt.show()



    def reset(self) -> None:
        self.gates : list[list[int | list[int]]] = []

        qr_nodes = QuantumRegister(self.log_num_nodes, 'q')
        qr_connections = QuantumRegister(self.log_num_connections, 'l')
        cr = ClassicalRegister(self.log_num_nodes, 'c')

        self.probabilities : list[list[float]] = []
        self.steps = 0
        self.qc = QuantumCircuit(qr_connections, qr_nodes, cr)


# TODO networkx bipartite draw function
class BiCollabFiltering(DiscreteTimeWalk):

    def __prepareCircuit(self, starting_parity : str) -> None:

        if starting_parity == "even":
            parity = 0
        else:
            parity = 1

        amplitudes = []

        num_possibilities = 2 ** (self.total_num_qubits)

        nodes_used = 0

        for i in range(num_possibilities):
            amplitudes.append(0)

        for node in range(self.num_nodes):

            inverse_of_sqrt_degree = 1/np.sqrt(self._DiscreteTimeWalk__nodes[node]["degree"])

            if node % 2 == parity:
                nodes_used += 1
                for connection in self._DiscreteTimeWalk__nodes[node]["connections"]:
                    binary_node = bin(node)[2:].zfill(self.log_num_nodes)
                    binary_connection = bin(connection)[2:].zfill(self.log_num_connections)
                    state = binary_node + binary_connection
                    amplitudes[int(state, 2)] = inverse_of_sqrt_degree

        inverse_of_sqrt_num_nodes_used = 1/np.sqrt(nodes_used)

        for i in range(num_possibilities):
            if amplitudes[i] != 0: # mathematically doesn't make a difference but is prittier to print and skip some divisions and multiplications
                amplitudes[i] *= inverse_of_sqrt_num_nodes_used

        self.qc.prepare_state(amplitudes)



    def simulate(self, steps : int, register_all_probabilities : bool, starting_node : int = -1, starting_parity : str = "none", state_prep_list : list[float] = []) -> None:

        if steps == 0:
            raise ValueError("steps must be a positive integer")

        if starting_node == -1 and starting_parity == "none" and state_prep_list == []:
            raise ValueError("Please choose a starting node or a parity, alternatively provide state preparation list")


        # Exclusivity handling
        if state_prep_list != [] and starting_node != -1:
            raise ValueError("starting_node and state_prep_list are mutually exclusive, please choose one")

        if state_prep_list != [] and starting_parity != "none":
            raise ValueError("starting_parity and state_prep_list are mutually exclusive, please choose one")

        if starting_parity != "none" and starting_node != -1:
            raise ValueError("starting_node and starting_parity are mutually exclusive, please choose one")



        # Decided to simply ignore negative numbers, might change it in the future
        if starting_node != -1 and starting_node >= 0:
            if starting_node > self.num_nodes-1:
                raise ValueError(f"starting_node must be between 0 and {self.num_nodes-1}")
            self._DiscreteTimeWalk__prepareCircuitFromStartingNode(starting_node)

        elif state_prep_list != []:
            if len(state_prep_list) != 2 ** (self.total_num_qubits):
                raise ValueError(f"state_prep_list must have {2 ** (self.total_num_qubits)} elements, one for each possible binary state")
            self.qc.prepare_state(state_prep_list)

        elif starting_parity != "even" and starting_parity != "odd":
            raise ValueError("starting_parity must be 'even' or 'odd'")



        else:
            self.__prepareCircuit(starting_parity)

        self.steps = steps

        for node in range(self.num_nodes):
            self._DiscreteTimeWalk__addCoin(node)

        for connection in range(self.num_connections):
            self._DiscreteTimeWalk__addShift(connection)

        # Following the 2x+1 equation
        for i in range(0, (2 * steps) + 1) :
            for gate in range(len(self.gates)):
                self.qc.append(self.gates[gate][0], self.gates[gate][1]) # Appending the gate (gates[gate][0]) to it's application list (gates[gate][1])

            # Only register the probabilities if i is one of the values of the equation's domain
            if register_all_probabilities and (i != 0 and i % 2 == 1):
                self._DiscreteTimeWalk__getProbabilities()

        if not register_all_probabilities:
            self._DiscreteTimeWalk__getProbabilities()


    # Really just a wrapper for the nx draw function, but it's good for simplicity and to maintain the lib self contained
    def draw(self, show_labels : bool = True) -> None:

        colors = []

        for i in range(0, int(np.floor(self.num_nodes/2))):
            colors.append("#d61c1c")
        for i in range(int(np.floor(self.num_nodes/2)), self.num_nodes):
            colors.append("#1f78b4")

        nx.draw(self.networkx_graph, with_labels=show_labels, node_color=colors)
        plt.show()

if __name__ == "__main__":

    study_matrix = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    simple_4x4 = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]

    lattice_5x5 = [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ]

    connected = [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]
    ]

    bipart = [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ]

    test = DiscreteTimeWalk(study_matrix)
    test.simulate(1, False)
    test.plotProbabilities(True)