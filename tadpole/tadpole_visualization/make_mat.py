import numpy, scipy.io
import pickle, sys

if (len(sys.argv) > 2):
    source_name = sys.argv[1]
    dest_name = sys.argv[2]

    a = pickle.load(open(source_name, "rb"))
    scipy.io.savemat(dest_name, mdict={'data': a})

    print(
        "Data successfully converted to .mat file with variable name \"data\"")
    sys.exit(0)
else:
    print(
        "Usage: pickle_to_mat_converter.py source_name.pickle mat_filename.mat"
    )
    sys.exit(1)
