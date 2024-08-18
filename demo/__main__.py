# Importing the benchmark will take a few seconds.
print("Loading the benchmark module...", end='', flush=True)
from cfro import CFRO
print("done.")

# Switch into the demo directory so relative path references work.
import os
CURR_DIR = os.sep.join(__file__.split(os.sep)[:-1])
os.chdir(CURR_DIR)

# See if the existing demo contents should be erased.
print("Enter y to erase the existing demo's data: ", end='')
erase_demo = bool(input() == 'y')
if erase_demo:
    import shutil
    shutil.rmtree('demo_data', ignore_errors=True)
    print('Erased.\n')
else:
    print()

# Load a benchmark pipeline with any parameters of interest.
benchmark = CFRO(
    label_faces=True,  # if True, labelling UI will be loaded
    names_path="names.csv",  # csv file with names of people in the dataset
    providers_path="../services.yaml",  # FRT providers credentials
    scraper_path="../scrapers.yaml",  # image scrapers credentials
    database_path="demo_data",  # folder name where outputs will be stored
    image_source="google_images",  # Options: "google_images", "google_news"
    max_number_of_photos=100,  # maximum number of photos to download per person
    subsampling_seed=2022,  # seed for subsampling cross-query pairs
    apply_majority_vote=True,  # if True, majority vote will be applied to estimate labels
    use_ramdisk_for_photos=True,  # if True, photos will be stored non-persistenly in RAM disk
)

# # Run the entire benchmark.
# benchmark.run()

# Alternatively, run the benchmark step by step.
benchmark.load_photos()

benchmark.detect_faces()

benchmark.compare_faces()

benchmark.label_detected_faces()

benchmark.analyze(
    create_pdf_report=True,  # creates detailed pdf report (takes a bit longer for large datasets)
    use_annotations=True,  # if True, uses annotations in results plots (if available), if False: ignores annotations
    fmr_fnmr_error_range=(0.0, 1.0)  # range of error rates to plot in FMR-FNMR plots
)
# All intermediate and final results are stored in the database_path folder.


benchmark.release_ramdisks()  # this only has an effect if use_ramdisk_for_photos=True

