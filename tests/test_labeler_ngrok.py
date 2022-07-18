import pytest
import os
import cfro

"""
Sample invokation: python3 -m tests.test_labeler_ngrok
"""

if __name__ == "__main__":
    # Bad practice but only way to prevent test failure...
    from .secret import NGROK_AUTH

    path = __file__.replace("test_labeler_ngrok.py", "tmp")

    database = cfro.Database(path)
    database._delete_database(force=True)
    database = cfro.Database(path, ngrok_token=NGROK_AUTH)

    person_names = ["Alec Baldwin", "Mitch McConnell", "Mark Zuckerberg"]
    for person_name in person_names:
        print("Adding person", person_name)
        person_id = database.add_person(person_name)
        version_id = database.add_version(
            person_id, download_source=cfro.ImageSource.GOOGLE_NEWS
        )
    dataset_id = database.add_dataset_with_latest_versions(
        list(range(len(person_names)))
    )
    database.update_provider_credentials(
        cfro.ProviderType.TEST_PROVIDER, {"pass": "password"}
    )
    benchmark_id = database.add_benchmark(dataset_id, [cfro.ProviderType.TEST_PROVIDER])
    tester = database.get_benchmark(0)
    tester.run_providers_detect()
    tester.label_detected_faces(ngrok=True)

    # Test how a clone is able to continue...

    database_clone = cfro.Database(path, ngrok_token=NGROK_AUTH)
    tester_clone = database_clone.get_benchmark(0)
    tester_clone.label_detected_faces(ngrok=True)
