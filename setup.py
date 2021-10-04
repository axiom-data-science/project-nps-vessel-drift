from pathlib import Path
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Vessel drift analysis'

setup(
    name='vessel_drift_analysis',
    version=VERSION,
    author='Jesse Lopez',
    author_email='jesse@axds.co',
    description=DESCRIPTION,
    packages=find_packages(),
    scripts=[str(p) for p in Path('scripts').glob('**/*.py')],
)