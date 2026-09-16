# Source this to get a python environment with uproot/awkward/vector and pyevtana on PYTHONPATH.
#   source /exp/mu2e/app/users/mmackenz/main/pyevtana/setup.sh
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv rootana 2.5.0
export PYTHONPATH="/exp/mu2e/app/users/mmackenz/main/pyevtana:${PYTHONPATH}"
