import heapq


class ResolvedFaces:
    """
    Represents a face resolved after using the IoU metric
    to group multiple :class:`DetectedFace`'s between
    multiple cloud providers.

    These are not persistent, rather recomputed upon database load.
    """

    def __init__(self, photo_id, person_id, faces):
        self.photo_id = photo_id
        self.person_id = person_id
        self.faces = faces


class FaceClusterer:
    """
    This class groups face detections from different providers.

    For a given photo, we take the union U of face detections over
    all providers. Then we compute all pairs of faces U x U.
    This allows us to create a graph with edge set U and edge set
    IoU(U x U) where IoU is the Intersection over Union function
    on two bounding boxes from face detections.

    Applying Kruskal's algorithm for hierarchical clustering to
    this graph outputs a list of face detections (at most one per
    provider) for each separate face in the input photo.
    """

    def __init__(self, provider_to_faces, threshold):
        # Map from provider to list of detected faces
        self.provider_to_faces = provider_to_faces
        # List of clusters
        self.clusters = []
        # Priority queue (iou, tiebreaker int, face, face)
        self.priority_q = []
        # List of all faces, union over providers
        self.faces = []
        # Threshold for IoU that determines if bounding boxes are equal
        self.threshold = threshold

    def compute_clusters(self):
        """
        Public interface to obtain the clusters of bounding boxes.
        """
        self._set_priority_queue_and_face_list()
        self._assemble_clusters()
        return self.clusters

    def _assemble_clusters(self):
        """
        Loads the clusters (as ResolvedFaces) into `self.clusters`
        """
        # Map from each face to the cluster it belongs to.
        face_id_to_cluster_id = {}
        # Keep the providers stored in each cluster to ensure
        # no cluster has multiple bounding boxes for a given provider.
        cluster_id_to_provider_set = {}

        # Iterate over each pair of overlapping bounding boxes,
        # in order of the IoU value (hence the priority queue).
        while len(self.priority_q) > 0:
            _, _, face1, face2 = heapq.heappop(self.priority_q)
            face1_has_cluster = face1.face_id in face_id_to_cluster_id
            face2_has_cluster = face2.face_id in face_id_to_cluster_id

            if not face1_has_cluster and not face2_has_cluster:
                # If neither face belongs to a cluster, create a new cluster.
                cluster_id = max(face_id_to_cluster_id.values() or [0]) + 1
                face_id_to_cluster_id[face1.face_id] = cluster_id
                face_id_to_cluster_id[face2.face_id] = cluster_id
                cluster_id_to_provider_set[cluster_id] = {
                    face1.provider,
                    face2.provider,
                }
                continue

            if face1_has_cluster and face2_has_cluster:
                # If both faces belong to (different) clusters,
                # combine the clusters, if there are no providers in common.
                cluster1_id = face_id_to_cluster_id[face1.face_id]
                cluster2_id = face_id_to_cluster_id[face2.face_id]
                if cluster1_id == cluster2_id:
                    continue
                providers1 = cluster_id_to_provider_set[cluster1_id]
                providers2 = cluster_id_to_provider_set[cluster2_id]
                if len(providers1.intersection(providers2)) == 0:
                    # Obtain a new cluster id and update each sub-cluster
                    # to reflect the new id in the mapping.
                    cluster_id = max(face_id_to_cluster_id.values() or [0]) + 1
                    for k, v in face_id_to_cluster_id.items():
                        if v in [cluster1_id, cluster2_id]:
                            face_id_to_cluster_id[k] = cluster_id
                    del cluster_id_to_provider_set[cluster1_id]
                    del cluster_id_to_provider_set[cluster2_id]
                    cluster_id_to_provider_set[cluster_id] = providers1.union(
                        providers2
                    )
                continue

            if face1_has_cluster and not face2_has_cluster:
                # Defer to the next case (which is symmetric).
                face1, face2 = face2, face1
                face1_has_cluster, face2_has_cluster = (
                    face2_has_cluster,
                    face1_has_cluster,
                )
            if not face1_has_cluster and face2_has_cluster:
                # Add face1 to the face2 cluster, as long as its provider isn't there.
                cluster_id = face_id_to_cluster_id[face2.face_id]
                if face1.provider not in cluster_id_to_provider_set[cluster_id]:
                    face_id_to_cluster_id[face1.face_id] = cluster_id
                    cluster_id_to_provider_set[cluster_id].add(face1.provider)

        for face in self.faces:
            if face.face_id not in face_id_to_cluster_id:
                # Any face not yet added to a cluster will form its own cluster.
                cluster_id = max(face_id_to_cluster_id.values() or [0]) + 1
                face_id_to_cluster_id[face.face_id] = cluster_id
                cluster_id_to_provider_set[cluster_id] = {face.provider}

        # Finally, we reformat the clusters into the output format.
        cluster_to_faces = {}
        for face in self.faces:
            cluster_id = face_id_to_cluster_id[face.face_id]
            if cluster_id not in cluster_to_faces:
                cluster_to_faces[cluster_id] = []
            cluster_to_faces[cluster_id].append(face)
        for faces in cluster_to_faces.values():
            self.clusters.append(
                ResolvedFaces(faces[0].photo_id, faces[0].person_id, faces)
            )

    def _set_priority_queue_and_face_list(self):
        """
        Sets a list of detected faces spanned by all providers.
        Sets a priority queue with pairs of faces, with priority
        given to bounding box pairs with higher IoU scores.
        """
        values = list(self.provider_to_faces.values())
        tiebreaker = 0
        # Loop over all pairs of faces with separate providers.
        for i in range(len(values)):
            for face1 in values[i]:
                self.faces.append(face1)
                for j in range(i + 1, len(values)):
                    for face2 in values[j]:
                        iou = face1.bounding_box.compute_intersection_over_union(
                            face2.bounding_box
                        )
                        # Only consider bounding boxes sufficiently close/overlapping.
                        if iou >= self.threshold:
                            # We use -iou because it is a min-heap and we want to
                            # simulate a max-heap.
                            q_elem = (-iou, tiebreaker, face1, face2)
                            self.priority_q.append(q_elem)
                            # Increment the tiebreaker index to ensure each value
                            # is unique (since the DetectedFace class does not
                            # have comparison implemented).
                            tiebreaker += 1
        # Convert the unordered list of priorities into a priority queue.
        heapq.heapify(self.priority_q)
