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
    label_faces=True,
    names_path="names.csv",
    providers_path="../services.yaml",
    scraper_path="../scrapers.yaml",
    database_path="demo_data",
    image_source="google_images",  # Options: "google_images", "google_news"
    max_number_of_photos=100,
    subsampling_seed=2022,
    apply_majority_vote=True
)

# # Run the entire benchmark.
# benchmark.run()

# Run the benchmark step by step.

benchmark.load_photos()

benchmark.detect_faces()

benchmark.compare_faces()

benchmark.label_detected_faces()

benchmark.analyze(
    create_pdf_report=True,  # creates detailed pdf report (takes a bit longer for large datasets)
    use_annotations=True,  # if True, uses annotations in results plots (if available), if False: ignores annotations
    fmr_fnmr_error_range=(0.0, 1.0)  # range of error rates to plot in FMR-FNMR plots
)