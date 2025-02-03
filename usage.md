### this is for running on windows adapt accordingly.

### make virtual env
py -3.12 -m venv venv

### activate venv
.\venv\Scripts\activate

### install requirements
pip install -r requirements.txt

### run Vanilla python (using scipy's minimize for BFGS)
py protein_folding_3d.py

### run cython version (using scipy's minimize for BFGS)
 py -m cython.protein_folding_3d

### run just cython version with implemented BFGS
 py -m cython.protein_folding_3d

