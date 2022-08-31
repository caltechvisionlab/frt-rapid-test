import numpy as np


def custom_1d_binary_kmeans(data: np.ndarray, fixed_assignments: np.ndarray, max_iter: int = 300, init_centroids: np.ndarray = None):
    if init_centroids is None:
        init_centroids = np.random.uniform(low=data.min(), high=data.max(), size=2)

    assignments = fixed_assignments.copy()
    centroids = init_centroids.copy()

    fixed_highest = data[fixed_assignments == 0].max()
    fixed_mean = data[fixed_assignments == 0].mean()
    smallest_allowed_positive_centroid = fixed_highest + abs(fixed_highest - fixed_mean)
    init_centroids[1] = max(init_centroids[1], smallest_allowed_positive_centroid)

    for _ in range(max_iter):
        # assign
        previous_assignments = assignments.copy()
        assign_indices = fixed_assignments == -1
        estimated_assignments = np.argmin(np.abs(data[:, None] - centroids), axis=1)
        assignments[assign_indices] = estimated_assignments[assign_indices]
        if np.all(assignments == previous_assignments):
            break
        # recompute centroids
        for i in [0, 1]:
            if not len(data[assignments == i]) == 0:
                new_cluster_mean = data[assignments == i].mean()
                if i == 0:
                    smallest_allowed_positive_centroid = fixed_highest + abs(fixed_highest - new_cluster_mean)
                if i == 1 and new_cluster_mean < smallest_allowed_positive_centroid:
                    new_cluster_mean = max(new_cluster_mean, smallest_allowed_positive_centroid)
                centroids[i] = new_cluster_mean

    return assignments, centroids




if __name__ == '__main__':
    x = np.array([0.75374612, 0.75466139, 0.73408848, 0.7402437 , 0.73312544,
       0.7400986 , 0.73868925, 0.77224042, 0.74434722, 0.75250142])
    y = np.array([-1, -1, -1, -1, -1, -1,  0,  0,  0,  0])
    inits = np.array([x[y==0].mean(), 1.])

    out = custom_1d_binary_kmeans(data=x, fixed_assignments=y, init_centroids=inits)
    print(out)