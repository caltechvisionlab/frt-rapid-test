import difPy
import networkx as nx
import os


# Don't run this directly -> use difpy.sh


def run_test(duplicate_photo_dir):
    search = difPy.dif(duplicate_photo_dir, similarity="low")

    # Construct an undirected graph where edges
    # represent duplicate images per difPy.
    G = nx.Graph()
    for d in search.result.values():
        orig = d["location"]
        dups = d["duplicates"]
        for dup in dups:
            G.add_edge(orig, dup)

    # Compute the connected components.
    cc = nx.connected_components(G)
    outcome = []
    for i, c in enumerate(cc):
        dups = set(f.split(os.sep)[-1] for f in c)
        outcome.append(dups)
    return outcome


def get_difpy_version():
    cmd = f"pip3 freeze | grep difPy"
    os.system(cmd)


cases = [
    ("tests/duplicates/set1", [{"temp_000017.jpg", "temp_000029.jpg"}]),
    ("tests/duplicates/set2", [{"temp_000002.jpg", "temp_000045.jpg"}]),
    ("tests/duplicates/set3", [{"temp_000010.jpg", "temp_000039.jpg"}]),
    (
        "tests/duplicates/set4",
        [
            {"temp_000034.jpg", "temp_000042.jpg"},
            {"temp_000004.jpg", "temp_000038.jpg"},
        ],
    ),
    ("tests/duplicates/set5", [{"temp_000017.jpg", "temp_000021.jpg"}]),
    ("tests/duplicates/set6", [{"temp_000021.jpg", "temp_000026.jpg"}]),
]
passes = 0
fails = 0
print("Starting tests:")
for test_set, expected in cases:
    output = run_test(test_set)
    if sorted(output) == sorted(expected):
        symbol = "=="
        passes += 1
    else:
        symbol = "!="
        fails += 1
    print("(Actual output)", output, symbol, expected, "(Expected output)")
print(f"Results: {passes} passes and {fails} fails\n")
