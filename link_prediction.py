from discrete_walk import * #temporary...?

class LinkPrediction():

    def __init__(self, graph : list[list[int]] | nx.Graph):
        self.dtw_simulation = DiscreteQuantumWalk(graph)
        self.scores : list[float] = []

    def predict(self, steps : int, register_all_scores : bool, starting_node : int) -> None:

        if steps == 0:
            raise ValueError("steps must be a positive integer")

        if starting_node > self.dtw_simulation.num_nodes-1:
            raise ValueError(f"starting_node must be between 0 and {self.dtw_simulation.num_nodes-1}")

        self.dtw_simulation.reset()
        self.dtw_simulation.simulate(
            steps=steps,
            register_probabilities=True,
            register_all_probabilities=register_all_scores,
            starting_node=starting_node
        )

        adjacent_nodes = self.dtw_simulation._DiscreteQuantumWalk__nodes[starting_node].connected_nodes

        starting_node_degree = self.dtw_simulation._DiscreteQuantumWalk__nodes[starting_node].degree

        scores = [0.0 for _ in self.dtw_simulation.probabilities[0]]

        for prob_list in self.dtw_simulation.probabilities:

            for node in range(len(prob_list)):
                if node in adjacent_nodes or node == starting_node:
                    continue

                # probability of node j * (degree of i + degree of j)
                scores[node] += prob_list[node] * (starting_node_degree + self.dtw_simulation._DiscreteQuantumWalk__nodes[node].degree)

        for i in range(len(scores)):
            scores[i] /= steps

        self.scores = scores

    def plotScores(self):
        plt.bar(range(len(self.scores)), self.scores)
        plt.show()

if __name__ == "__main__":
    simple_4x4 = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]

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
    # função que dá a probabilidade de estar no node x ao longo do tempo
    lp = LinkPrediction(study_matrix)
    lp.predict(steps=2, register_all_scores=True, starting_node=5)
    lp.plotScores()
    lp.dtw_simulation.draw()
    #a = DiscreteQuantumWalk(study_matrix)
    #a.reset()
    #a.simulate(3, True, False, 5)
    #a.plotProbabilities(False)
    #print(a.probabilities)