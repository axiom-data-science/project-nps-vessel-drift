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
    scripts=['scripts/calculate_drift_hazard.py']
)