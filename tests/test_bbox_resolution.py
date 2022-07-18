import pytest
import os
import cfro
import numpy as np

"""
Sample invokation: python3 -m pytest tests/test_bbox_resolution.py

Check IoU metric is correct, and that faces are resolved
across providers as expected (e.g., for a multi-face image).
"""

bbox = cfro.BoundingBox(0, 0, 100, 100)


def test_bbox_against_self():
    assert bbox.compute_intersection_over_union(bbox) == 1


def test_bbox_shifted_right_100():
    bbox_shifted_right_100 = cfro.BoundingBox(100, 0, 200, 100)
    assert bbox.compute_intersection_over_union(bbox_shifted_right_100) == 0
    assert bbox_shifted_right_100.compute_intersection_over_union(bbox) == 0


def test_bbox_shifted_right_200():
    bbox_shifted_right_200 = cfro.BoundingBox(200, 0, 300, 100)
    assert bbox.compute_intersection_over_union(bbox_shifted_right_200) == 0
    assert bbox_shifted_right_200.compute_intersection_over_union(bbox) == 0


def test_bbox_shifted_right_50():
    bbox_shifted_right_50 = cfro.BoundingBox(50, 0, 150, 100)
    assert bbox.compute_intersection_over_union(bbox_shifted_right_50) == 1 / 3
    assert bbox_shifted_right_50.compute_intersection_over_union(bbox) == 1 / 3


def test_bbox_shifted_up_100():
    bbox_shifted_up_100 = cfro.BoundingBox(0, 100, 100, 200)
    assert bbox.compute_intersection_over_union(bbox_shifted_up_100) == 0
    assert bbox_shifted_up_100.compute_intersection_over_union(bbox) == 0


def test_bbox_shifted_up_200():
    bbox_shifted_up_200 = cfro.BoundingBox(0, 200, 100, 300)
    assert bbox.compute_intersection_over_union(bbox_shifted_up_200) == 0
    assert bbox_shifted_up_200.compute_intersection_over_union(bbox) == 0


def test_bbox_shifted_up_50():
    bbox_shifted_up_50 = cfro.BoundingBox(0, 50, 100, 150)
    assert bbox.compute_intersection_over_union(bbox_shifted_up_50) == 1 / 3
    assert bbox_shifted_up_50.compute_intersection_over_union(bbox) == 1 / 3


def test_bbox_shifted_up_50_right_50():
    bbox_shifted_up_50_right_50 = cfro.BoundingBox(50, 50, 150, 150)
    assert bbox.compute_intersection_over_union(bbox_shifted_up_50_right_50) == 1 / 7
    assert bbox_shifted_up_50_right_50.compute_intersection_over_union(bbox) == 1 / 7


def test_face_clusterer():
    # These two faces overlap with high IoU and belong to separate providers.
    face1_cluster1 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(0, 0, 100, 100),
        cfro.ProviderType.TEST_PROVIDER,
        None,
    )
    face2_cluster1 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(5, 5, 105, 105),
        cfro.ProviderType.TEST_PROVIDER_V2,
        None,
    )

    # These faces also overlap with high IoU and belong to separate providers.
    face3_cluster2 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(200, 200, 300, 300),
        cfro.ProviderType.TEST_PROVIDER_V2,
        None,
    )
    face4_cluster2 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(201, 201, 301, 301),
        cfro.ProviderType.MICROSOFT_AZURE,
        None,
    )

    # This face belongs to a different provider than the two above,
    # but it does not have a sufficiently high IoU despite having some overlap.
    face5_cluster3 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(250, 250, 350, 350),
        cfro.ProviderType.TEST_PROVIDER,
        None,
    )

    # This face belongs to the same provider as `face4_cluster2` and has
    # slightly less overlap with `face3_cluster2` so it is invalid.
    face6_cluster4 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(202, 202, 302, 302),
        cfro.ProviderType.MICROSOFT_AZURE,
        None,
    )

    face1_cluster1.set_face_id(1)
    face2_cluster1.set_face_id(2)
    face3_cluster2.set_face_id(3)
    face4_cluster2.set_face_id(4)
    face5_cluster3.set_face_id(5)
    face6_cluster4.set_face_id(6)

    provider_to_faces = {
        cfro.ProviderType.TEST_PROVIDER: [
            face1_cluster1,
            face5_cluster3,
        ],
        cfro.ProviderType.TEST_PROVIDER_V2: [
            face2_cluster1,
            face3_cluster2,
        ],
        cfro.ProviderType.MICROSOFT_AZURE: [
            face4_cluster2,
            face6_cluster4,
        ],
    }

    clusterer = cfro.FaceClusterer(provider_to_faces, 0.2)

    clusters = clusterer.compute_clusters()
    cluster_set = [
        list(sorted(f.face_id for f in resolved_faces.faces))
        for resolved_faces in clusters
    ]
    cluster_set.sort()
    assert cluster_set == [
        [
            1,
            2,
        ],
        [
            3,
            4,
        ],
        [
            5,
        ],
        [
            6,
        ],
    ]


def test_face_clusterer_v2():
    # These four faces overlap with high IoU and belong to separate providers.
    #
    face1_cluster1 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(0, 0, 100, 100),
        cfro.ProviderType.TEST_PROVIDER,
        None,
    )
    face2_cluster1 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(1, 1, 101, 101),
        cfro.ProviderType.TEST_PROVIDER_V2,
        None,
    )
    face3_cluster1 = cfro.DetectedFace(
        None,
        None,
        cfro.BoundingBox(4, 4, 104, 104),
        cfro.ProviderType.MICROSOFT_AZURE,
        None,
    )
    face4_cluster1 = cfro.DetectedFace(
        None, None, cfro.BoundingBox(5, 5, 105, 105), cfro.ProviderType.AMAZON_AWS, None
    )

    face1_cluster1.set_face_id(1)
    face2_cluster1.set_face_id(2)
    face3_cluster1.set_face_id(3)
    face4_cluster1.set_face_id(4)

    provider_to_faces = {
        cfro.ProviderType.TEST_PROVIDER: [
            face1_cluster1,
        ],
        cfro.ProviderType.TEST_PROVIDER_V2: [
            face2_cluster1,
        ],
        cfro.ProviderType.MICROSOFT_AZURE: [
            face3_cluster1,
        ],
        cfro.ProviderType.AMAZON_AWS: [
            face4_cluster1,
        ],
    }

    clusterer = cfro.FaceClusterer(provider_to_faces, 0.2)

    clusters = clusterer.compute_clusters()
    cluster_set = [
        list(sorted(f.face_id for f in resolved_faces.faces))
        for resolved_faces in clusters
    ]
    cluster_set.sort()
    assert cluster_set == [
        [
            1,
            2,
            3,
            4,
        ]
    ]
