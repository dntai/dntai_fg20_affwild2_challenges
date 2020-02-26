# config root dir, library path
import os, sys
root_dir      = os.path.abspath('../').replace("\\", "/")
source_dir    = os.path.join(root_dir, "submit", "sources").replace("\\", "/")
include_dirs  = [source_dir]
for lib in include_dirs:
    if lib not in sys.path: sys.path.insert(0, lib)
# np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.4f}'.format}, linewidth=1000)

# common info of project
data_dir       = os.path.join(root_dir, "data").replace("\\", "/")
dataset_dir    = os.path.join(data_dir, "datasets").replace("\\", "/")
sub_dir        = os.path.join(root_dir, "submit").replace("\\", "/")

# path of module
module_dir        = os.path.abspath(".").replace("\\", "/")
relate_module_dir = os.path.relpath(module_dir, start=source_dir).replace("\\", "/")
