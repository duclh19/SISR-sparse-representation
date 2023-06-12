import os 
import zipfile
from pathlib import Path
def unzip(file, to_location):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(to_location)

path = Path('.')
unzip('Set5.zip', path/'data')
unzip('T91.zip', path/'data')

# !mv SISR-sparse-representation/data/Set5/original SISR-sparse-representation/data/set5_hr
# !rm -r SISR-sparse-representation/data/Set5/