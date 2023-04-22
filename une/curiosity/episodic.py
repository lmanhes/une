from collections import deque
from typing import NamedTuple, Dict

from loguru import logger
import torch
import torch.nn as nn

from une.memories.utils.running_stats import RunningStats


class KNNQueryResult(NamedTuple):
    neighbors: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_distances: torch.Tensor


def _cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Returns the squared Euclidean distance between the two inputs.
    A simple equation on Euclidean distance for one-dimension:
        d(p,q)=sqrt[(p-q)^2]
    """
    return torch.sum(torch.square(a - b))


def knn_query(data: torch.Tensor, memory: torch.Tensor, num_neighbors: int) -> KNNQueryResult:
    """Finds closest neighbors in data to the query points & their squared euclidean distances.
    Args:
      data: tensor of embedding, shape [embedding_size].
      memory: tensor of previous embedded data, shape [m, embedding_size],
        where m is the number of previous embeddings.
      num_neighbors: number of neighbors to find.
    Returns:
      KNNQueryResult with (all sorted by squared euclidean distance):
        - neighbors, shape [num_neighbors, feature size].
        - neighbor_indices, shape [num_neighbors].
        - neighbor_distances, shape [num_neighbors].
    """
    assert memory.shape[0] >= num_neighbors

    distances = torch.stack([_cdist(memory[i], data) for i in range(memory.shape[0])], dim=0)

    distances, indices = distances.topk(num_neighbors, largest=False)
    neighbors = torch.stack([memory[i] for i in indices], dim=0)
    return KNNQueryResult(neighbors=neighbors, neighbor_indices=indices, neighbor_distances=distances)


class EpisodicCuriosityModule(object):
    def __init__(
        self,
        encoder: nn.Module,
        memory_size: int,
        k: int,
        kernel_epsilon: float = 0.0001,
        cluster_distance: float = 0.008,
        max_similarity: float = 8.0,
        c_constant: float = 0.001,
    ) -> None:
        self.encoder = encoder
        self.k = k
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.max_similarity = max_similarity
        self.c_constant = c_constant
        self.episodic_memory = deque(maxlen=memory_size)
        self.euclidean_dist_running_stats = RunningStats()

    def get_episodic_reward(self, observation: torch.Tensor):
        with torch.no_grad():
            embedding = self.encoder(observation).squeeze(0)

            # Insert single embedding into memory.
            self.episodic_memory.append(embedding)

            memory = list(self.episodic_memory)
            if len(memory) < self.k:
                return 0.0
            
            memory = torch.stack(memory, dim=0)
            knn_query_result = knn_query(embedding, memory, self.k)
            # neighbor_distances from knn_query is the squared Euclidean distances.
            nn_distances_sq = knn_query_result.neighbor_distances

            self.euclidean_dist_running_stats += nn_distances_sq.cpu().numpy()

            # Normalize distances with running mean dₘ².
            distance_rate = nn_distances_sq / self.euclidean_dist_running_stats.mean

            # The distance rate becomes 0 if already small: r <- max(r-ξ, 0).
            distance_rate = torch.min((distance_rate - self.cluster_distance), torch.tensor(0.0))

            # Compute the Kernel value K(xₖ, x) = ε/(rate + ε).
            kernel_output = self.kernel_epsilon / (distance_rate + self.kernel_epsilon)

            # Compute the similarity for the embedding x:
            # s = √(Σ_{xₖ ∈ Nₖ} K(xₖ, x)) + c
            similarity = torch.sqrt(torch.sum(kernel_output)) + self.c_constant

            if torch.isnan(similarity):
                return 0.0

            # Compute the intrinsic reward:
            # r = 1 / s.
            if similarity > self.max_similarity:
                return 0.0

            return (1 / similarity).cpu().item()

    def reset(self) -> None:
        self.episodic_memory.clear()
