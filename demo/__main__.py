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
    providers_path="providers.csv",
    constants_config_path="config.ini",
    database_path="demo_data",
    max_number_of_photos=50,
    subsampling_seed=2022,
)

# # Run the entire benchmark.
# benchmark.run()

# Run the benchmark step by step.
benchmark.load_photos()
benchmark.detect_faces()
benchmark.label_detected_faces()
benchmark.compare_faces()
benchmark.analyze()