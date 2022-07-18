import pytest
import os
import cfro

"""
Sample invokation: python3 -m pytest tests/test_workflow.py -v
"""


def test_basic_import():
    assert cfro.benchmark_version == "1.0.0"


def test_basic_workflow(tmpdir):
    """
    Tests a basic end-to-end workflow.
    A temporary db path is created using this method:
        https://docs.pytest.org/en/6.2.x/tmpdir.html
    """
    # Create a new database.
    tmp_db_path = str(tmpdir.mkdir("db"))

    sample_upload_path = __file__.replace("test_workflow.py", "sample_images")
    num_uploads = len(os.listdir(sample_upload_path))

    database = cfro.Database(tmp_db_path)

    assert database.get_num_photos() == 0

    # Load some people into the database.
    # Ensure the person_id scheme is correct.
    person_names = ["Jeff Bezos", "Elon Musk", "Ethan Mann"]
    for true_id, person_name in enumerate(person_names):
        gen_id = database.add_person(person_name)
        assert gen_id == true_id

        output_id = database.get_person_id(person_name)
        assert output_id == true_id

        person = database.get_person(true_id)
        assert person.get_name() == person_name

        # Only download images for the 1st person to save time.
        if true_id == 0:
            version_id1 = database.add_version(
                true_id,
                download_source=cfro.ImageSource.GOOGLE_NEWS,
                max_number_of_photos=20,
            )
            assert version_id1 == 0

        # But include local uploads for everyone.
        version_id2 = database.add_version(true_id, upload_path=sample_upload_path)
        assert version_id2 == (1 if true_id == 0 else 0)

    dataset_id = database.add_dataset_with_latest_versions([0, 2])
    assert dataset_id == 0

    num_dataset_photos = 2 * num_uploads

    # TODO - spreadsheet of credentials or JSON
    database.update_provider_credentials(
        cfro.ProviderType.TEST_PROVIDER, {"pass": "password"}
    )

    benchmark_id = database.add_benchmark(dataset_id, [cfro.ProviderType.TEST_PROVIDER])
    assert benchmark_id == 0

    tester = database.get_benchmark(0)
    assert tester.providers_enums == [cfro.ProviderType.TEST_PROVIDER]
    tester.set_constants_config({"INTERSECTION_OVER_UNION_IOU_THRESHOLD": 0.2})

    tester.run_providers_detect()
    test_provider = tester.providers[0]
    test_detected_faces = test_provider.detected_faces
    assert len(test_detected_faces.values()) == num_dataset_photos
    test_sub_detected_faces = list(test_detected_faces.values())[0]
    assert test_sub_detected_faces[0].provider == cfro.ProviderType.TEST_PROVIDER
    assert test_sub_detected_faces[0].metadata == "This is metadata."

    tester.label_detected_faces(bypass=True)
    tester.run_providers_compare()

    num_detected_faces = num_dataset_photos
    assert (
        len(test_provider.comparisons)
        == num_detected_faces * (num_detected_faces - 1) // 2
    )
    for ((face1_id, _), (face2_id, _)), confidence in test_provider.comparisons.items():
        assert confidence == eval(f"0.{face1_id}{face2_id}")

    # Open a clone of the database to ensure the data is accurate.
    database_clone = cfro.Database(tmp_db_path)
    for true_id, person_name in enumerate(person_names):
        output_id = database_clone.get_person_id(person_name)
        assert output_id == true_id

        person = database_clone.get_person(true_id)
        assert person.get_name() == person_name

        assert person.get_num_versions() == 2 if true_id == 0 else 1

        # TODO - This part can be improved, with a better interface.
        min_download_threshold = 5
        if true_id == 0:
            assert len(person.versions[0].get_photos()) > min_download_threshold
        assert len(person.versions[-1].get_photos()) == num_uploads

    dataset = database_clone.get_dataset(0)
    assert dataset.get_person_and_version_ids() == [(0, 1), (2, 0)]

    creds = database_clone.credentials[cfro.ProviderType.TEST_PROVIDER]
    assert creds["pass"] == "password"

    tester_clone = database_clone.get_benchmark(0)
    assert tester_clone.providers_enums == [cfro.ProviderType.TEST_PROVIDER]
    tester_clone.set_constants_config({"INTERSECTION_OVER_UNION_IOU_THRESHOLD": 0.2})

    test_provider = tester_clone.providers[0]
    test_detected_faces = test_provider.detected_faces
    assert len(test_detected_faces.values()) == num_dataset_photos
    test_sub_detected_faces = list(test_detected_faces.values())[0]
    assert test_sub_detected_faces[0].provider == cfro.ProviderType.TEST_PROVIDER
    assert test_sub_detected_faces[0].metadata == "This is metadata."

    num_detected_faces = num_dataset_photos
    assert (
        len(test_provider.comparisons)
        == num_detected_faces * (num_detected_faces - 1) // 2
    )
    for ((face1_id, _), (face2_id, _)), confidence in test_provider.comparisons.items():
        assert confidence == eval(f"0.{face1_id}{face2_id}")

    # Hard-reset the database.
    database._delete_database(force=True)
    empty_database = cfro.Database(tmp_db_path)
    assert empty_database.get_num_photos() == 0
