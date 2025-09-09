from qiskit import QuantumCircuit
from qiskit.circuit.library.standard_gates import XGate
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.visualization import plot_histogram

from qiskit.quantum_info import Statevector

def getGrade(node_index):
    grade = 0
    for i in range(0, mat_size):
        if adj_matrix[node_index][i]:
            grade += 1
    
    return grade

def isConnected(node_index, connection_id):

    for i in range(0, mat_size):
        if matrix_ids[node_index][i] == connection_id:
            return adj_matrix[node_index][i]
    return 0

def getAmplitude(node_index):

    grade = getGrade(node_index)
    amplitudes = []
    for i in range(0, num_connections): # Going through all the possibilities
        if isConnected(node_index, i):
            amplitudes.append(1/np.sqrt(grade))
        else:
            amplitudes.append(0)

    return amplitudes
        
def addCoin(node_index):

    binary = bin(node_index)[2:].zfill(log_size) # Getting the binary representation of the numbers with the correct amount of bits because python cuts them for some reason

    sub_qc = QuantumCircuit(log_num_connections)

    amplitudes = getAmplitude(node_index)

    # Apply Si dagger
    sub_qc.prepare_state(amplitudes)

    for i in range(0, log_num_connections):
        sub_qc.x(i)

    sub_qc.h(log_num_connections-1)
    sub_qc.append(CXGate, sub_control_list_coin)
    sub_qc.h(log_num_connections-1)

    for i in range(0, log_num_connections):
        sub_qc.x(i)

    # Apply Si
    sub_qc.prepare_state(amplitudes).inverse()

    coin_gate = sub_qc.to_gate().control(log_size, label=f"C{node_index}", ctrl_state=binary[::-1]) # Reversing the binary to make it identical to the research paper

    qc.append(coin_gate, control_list_coin)

def addShift(connection_id):

    binary = bin(connection_id)[2:].zfill(log_num_connections)

    sub_qc = QuantumCircuit(log_size)

    found = 0

    for i in range(0, mat_size):
        if found:
            break
        for j in range(0, mat_size):
            if matrix_ids[i][j] == connection_id:
                row = i
                column = j
                found = 1
                break

    binary_row = bin(row)[2:].zfill(log_size)
    binary_column = bin(column)[2:].zfill(log_size)

    for i in range(0, log_size):
        if binary_column[i] != binary_row[i]:
            sub_qc.x(i)

    shift_gate = sub_qc.to_gate().control(log_num_connections, label=f"S{connection_id}", ctrl_state=binary[::-1])

    qc.append(shift_gate, control_list_shift) # Inverted so the controls are on the bottom

def prepare():

    amplitudes = []

    for i in range(0, mat_size * num_connections):
            amplitudes.append(0)

    for node in range(0, mat_size):
        for connection in range(0, num_connections):
            if (isConnected(node, connection)):
                
                node_binary = bin(node)[2:].zfill(log_size)
                connection_binary = bin(connection)[2:].zfill(log_num_connections)
                full_binary = node_binary + connection_binary

                print(node_binary + " " + connection_binary)
                index = int(full_binary, 2)
                amplitudes[index] = 1/np.sqrt(getGrade(node))


    sqrt_n = np.sqrt(mat_size)

    print(amplitudes)

    for i in range(0, mat_size * num_connections):
        amplitudes[i] *= 1/sqrt_n

    print("\n\n")
    print(amplitudes)
    qc.prepare_state(amplitudes)

if __name__ == "__main__":

    # ===================================
    # Input
    # ===================================

    mat_size = int(input("Adjacency matrix's order: "))

    log_size = int(np.ceil(np.log2(mat_size))) # Will be used to get the binary representation of a number

    num_connections = 0

    adj_matrix = []

    matrix_ids = []
    
    #adj_matrix = [
    #    [0, 1, 1, 1],
    #    [1, 0, 1, 0],
    #    [1, 1, 0, 0],
    #    [1, 0, 0, 0]
    #]
#
    #matrix_ids = [
    #    [0, 1, 1, 1],
    #    [1, 0, 1, 0],
    #    [1, 1, 0, 0],
    #    [1, 0, 0, 0]
    #]

    adj_matrix = [
    [0, 0, 1, 0, 0, 0, 0, 1],  # vértice 0
    [0, 0, 0, 0, 1, 0, 0, 0],  # vértice 1
    [1, 0, 0, 0, 1, 0, 1, 0],  # vértice 2
    [0, 0, 0, 0, 1, 1, 0, 0],  # vértice 3
    [0, 1, 1, 1, 0, 0, 0, 0],  # vértice 4
    [0, 0, 0, 1, 0, 0, 0, 0],  # vértice 5
    [0, 0, 1, 0, 0, 0, 0, 1],  # vértice 6
    [1, 0, 0, 0, 0, 0, 1, 0]   # vértice 7
    ]

    matrix_ids = [
    [0, 0, 1, 0, 0, 0, 0, 1],  # vértice 0
    [0, 0, 0, 0, 1, 0, 0, 0],  # vértice 1
    [1, 0, 0, 0, 1, 0, 1, 0],  # vértice 2
    [0, 0, 0, 0, 1, 1, 0, 0],  # vértice 3
    [0, 1, 1, 1, 0, 0, 0, 0],  # vértice 4
    [0, 0, 0, 1, 0, 0, 0, 0],  # vértice 5
    [0, 0, 1, 0, 0, 0, 0, 1],  # vértice 6
    [1, 0, 0, 0, 0, 0, 1, 0]   # vértice 7
    ]
    # Reading the adjacency matrix
    #for i in range(0, mat_size):
    #    row_str = input(f"row {i+1} (space separated): ")
    #    row_elements = list(map(int, row_str.split())) # This is horrible to read but works flawlessly
    #    adj_matrix.append(row_elements)
    #    matrix_ids.append(row_elements)

    adj_matrix = np.array(adj_matrix)

    matrix_ids = np.array(matrix_ids)
    
    # Giving the ids
    for i in range(0, mat_size):
        for j in range(0, mat_size):
            matrix_ids[i][j] = -1
    
    for i in range(0, mat_size):
        for j in range(0, mat_size):
            if adj_matrix[i][j] == 1 and i < j:
                matrix_ids[i][j] = num_connections
                matrix_ids[j][i] = num_connections
                num_connections += 1

    log_num_connections = int(np.ceil(np.log2(num_connections)))

    # ===================================
    # Creating the Circuit and running it
    # ===================================

    qc = QuantumCircuit(log_size + log_num_connections, log_size)

    #prepare()

    control_list_coin = range(log_size + log_num_connections)
    sub_control_list_coin = range(log_num_connections) # List of the controls to the CX gate inside the coin

    control_list_shift = []
    print(matrix_ids)
    # Doing all this so the first items of the list are the connections's qubits (control) and the other ones are the node's qubits (affected)
    for i in range(0, log_num_connections):
        control_list_shift.append(i + log_size)
    for i in range(0, log_size):
        control_list_shift.append(i)

    CXGate = XGate().control(log_num_connections-1)

    for i in range(0, mat_size):
        addCoin(i)

    for i in range(0, num_connections):
        addShift(i)

    #qc.measure(range(log_size), range(log_size))

    #qc.draw(output="mpl")
    #plt.show()

    #backend = StatevectorSimulator()

    #transpiled_circuit = transpile(qc, backend)
    #transpiled_circuit.draw('mpl')
    #plt.show()
    #job = backend.run(backend)
    #print(job.result())
    #counts = job.result().get_counts()
    #plt.bar(list(counts.keys()), list(counts.values()))
    #plt.show()
    #print(counts)
    statevector = Statevector(qc)
    print(statevector)
    
    probs = []
    
    for node in range(0, mat_size):
        prob = 0
        for connection in range(0, num_connections):
            if (isConnected(node, connection)):
                
                node_binary = bin(node)[2:].zfill(log_size)
                connection_binary = bin(connection)[2:].zfill(log_num_connections)
                full_binary = node_binary + connection_binary

                print(node_binary + " " + connection_binary)
                index = int(full_binary, 2)
                #amplitudes[index] = 1/np.sqrt(getGrade(node))
                prob += abs(statevector[index]) ** 2
        probs.append(prob)

    plt.bar(list(range(mat_size)), probs)
    plt.show()
                