# Usage: from root, run `bash tests/difpy.sh`

# Silently install difPy v2.2
pip3 install difPy==2.2 > /dev/null
# Confirm version
pip3 freeze | grep difPy
# Run test cases
python3 -m tests.test_difpy

# Silently install difPy v2.3
pip3 install difPy==2.3 > /dev/null
# Confirm version
pip3 freeze | grep difPy
# Run test cases
python3 -m tests.test_difpy

# Silently install difPy v2.4
pip3 install difPy==2.4 > /dev/null
# Confirm version
pip3 freeze | grep difPy
# Run test cases
python3 -m tests.test_difpy

# Silently install difPy v2.4.1
pip3 install difPy==2.4.1 > /dev/null
# Confirm version
pip3 freeze | grep difPy
# Run test cases
python3 -m tests.test_difpy