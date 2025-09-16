from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library.standard_gates import XGate
import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator

from qiskit.quantum_info import Statevector
def prepareCircuit():

    amplitudes = []

    num_possibilities = 2 ** (log_num_nodes + log_num_connections)

    for i in range(num_possibilities):
        amplitudes.append(0)

    for node in range(num_nodes):

        inverse_of_sqrt_degree = 1/np.sqrt(nodes[node]["degree"])

        for connection in nodes[node]["connections"]:
            binary_node = bin(node)[2:].zfill(log_num_nodes)
            binary_connection = bin(connection)[2:].zfill(log_num_connections)

            #print(binary_node + " " + binary_connection)
            index = int(binary_node + binary_connection, 2)
            amplitudes[index] = inverse_of_sqrt_degree

    inverse_of_sqrt_num_nodes = 1/np.sqrt(num_nodes)

    for i in range(num_possibilities):
        if amplitudes[i] != 0: # mathematically doesn't make a difference but is prittier to print and skip some divisions and multiplications
            amplitudes[i] *= inverse_of_sqrt_num_nodes

    qc.prepare_state(amplitudes)

def getAmplitudeOfNode(node):

    amplitudes = []

    inverse_of_sqrt_degree = 1/np.sqrt(nodes[node]["degree"])

    for i in range(2 ** log_num_connections):
        amplitudes.append(0)

    for connection in nodes[node]["connections"]:
        amplitudes[connection] = inverse_of_sqrt_degree

    return amplitudes

def addCoin(node):

    binary_node = bin(node)[2:].zfill(log_num_nodes)

    sub_qc_coin = QuantumCircuit(log_num_connections)

    amplitudes = getAmplitudeOfNode(node)

    # Apply Si dagger
    sub_qc_coin.prepare_state(amplitudes[::-1]).inverse()

    for i in range(0, log_num_connections):
        sub_qc_coin.x(i)

    sub_qc_coin.h(log_num_connections-1)
    sub_qc_coin.append(CXGate, application_list_CXGate)
    sub_qc_coin.h(log_num_connections-1)

    for i in range(0, log_num_connections):
        sub_qc_coin.x(i)

    # Apply Si
    sub_qc_coin.prepare_state(amplitudes)

    coin_gate = sub_qc_coin.to_gate().control(log_num_nodes, label=f"C{node}", ctrl_state=binary_node[::-1]) # Reversing the binary to make it identical to the research paper

    gates.append([coin_gate, application_list_coin])

    qc.append(coin_gate, application_list_coin)

def addShift(connection):

    binary_connection = bin(connection)[2:].zfill(log_num_connections)

    sub_qc_shift = QuantumCircuit(log_num_nodes)

    binary_node1 = bin(connections[connection]["node1"])[2:].zfill(log_num_nodes)
    binary_node2 = bin(connections[connection]["node2"])[2:].zfill(log_num_nodes)

    for qubit in range(log_num_nodes):
        if binary_node1[qubit] != binary_node2[qubit]:
            sub_qc_shift.x(qubit)

    shift_gate = sub_qc_shift.to_gate().control(log_num_connections, label=f"S{connection}", ctrl_state=binary_connection[::-1])

    gates.append([shift_gate, application_list_shift])

    qc.append(shift_gate, application_list_shift)

if __name__ == "__main__":

    # ===================================
    # Input and boilerplate
    # ===================================

    num_nodes = 8
    adj_matrix = [
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0]
    ]

    #num_nodes = 4
    #adj_matrix = [
    #    [0, 1, 1, 1],
    #    [1, 0, 1, 0],
    #    [1, 1, 0, 0],
    #    [1, 0, 0, 0]
    #]

    steps = 1 #int(input("How many steps should the walk simulate?: "))

    #adj_matrix = np.array(adj_matrix)

    nodes = {}

    for i in range(num_nodes):
        nodes[i] = {"degree": 0, "connections": []}

    connections = {}

    num_connections = 0

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (adj_matrix[i][j]):
                connections[num_connections] = {"node1": i, "node2": j}
                nodes[i]["degree"] += 1
                nodes[i]["connections"].append(num_connections)
                nodes[j]["degree"] += 1
                nodes[j]["connections"].append(num_connections)
                num_connections += 1

    log_num_nodes = int(np.ceil(np.log2(num_nodes)))

    log_num_connections = int(np.ceil(np.log2(num_connections)))

    CXGate = XGate().control(log_num_connections-1)

    application_list_CXGate = list(range(log_num_connections))

    application_list_coin = list(range(log_num_nodes + log_num_connections))

    application_list_shift = list(range(log_num_nodes, log_num_connections + log_num_nodes)) + list(range(log_num_nodes)) # Control qubits have to be the first ones

    # ===================================
    # Making the circuit
    # ===================================

    qr_nodes = QuantumRegister(log_num_nodes, 'q')
    qr_connections = QuantumRegister(log_num_connections, 'l')
    cr = ClassicalRegister(log_num_nodes, 'c')

    qc = QuantumCircuit(qr_nodes, qr_connections, cr)

    prepareCircuit()

    gates = []

    for node in range(num_nodes):
        addCoin(node)

    #010011

    for connection in range(num_connections):
        addShift(connection)

    # Stored the gates so when running big circuits it avoids wasteful repeating of gate calculations
    #for step in range(1, steps):
    #    for gate in range(len(gates)):
    #        qc.append(gates[gate][0], gates[gate][1])

    #qc.measure(qr_connections, cr)

    """ simulator = AerSimulator()


    compiled = transpile(qc, simulator)
    job = simulator.run(compiled, shots=100000)
    result = job.result()
    counts = result.get_counts()

    myKeys = list(counts.keys())
    myKeys.sort()

    # Sorted Dictionary
    sd = {i: counts[i] for i in myKeys}

    plt.bar(range(len(sd)), list(sd.values()), align='center')
    plt.xticks(range(len(sd)), list(sd.keys())) """

    statevector = Statevector(qc)
    #print(statevector)
    
    asa = statevector.probabilities_dict()
    probs = []

    for node in range(num_nodes):
        prob = 0
        for connection in nodes[node]["connections"]:
            binary_node = bin(node)[2:].zfill(log_num_nodes)
            binary_connection = bin(connection)[2:].zfill(log_num_connections)

            #print(binary_node + " " + binary_connection)
            a = binary_node + binary_connection
            index = int(a, 2)
            #amplitudes[index] = inverse_of_sqrt_degree
            #print(a, abs(statevector[index] ** 2))
            prob += asa[a[::-1]] if a[::-1] in asa else 0
        probs.append(prob)
    plt.bar(range(num_nodes), probs)
    plt.show()

    myKeys = list(asa.keys())
    myKeys.sort()

    # Sorted Dictionary
    sd = {i: asa[i] for i in myKeys}

    plt.bar(range(len(sd)), list(sd.values()), align='center')
    plt.xticks(range(len(sd)), list(sd.keys()))
    print(asa)
    qc.draw(output="mpl")
    plt.show()