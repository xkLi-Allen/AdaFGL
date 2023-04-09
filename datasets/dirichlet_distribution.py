import numpy as np
from config import args

def dirichlet_distribution(num_entity, num_assignments, alpha=args.dirichlet_alpha):
    entity_idx = [x for x in range(num_entity)]
    idx_per_client = [[]] * num_assignments
    np.random.shuffle(entity_idx)
    proportions = np.random.dirichlet(np.repeat(alpha, num_assignments))
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(entity_idx)).astype(int)[:-1]
    idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(entity_idx, proportions))]
    entity_dict = {i:idx_per_client[i] for i in range(num_assignments)}
    return entity_dict