from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'inflection',
    'matplotlib',
    'python-dateutil',
    'requests',
    'tensorflow',
    'scikit-learn',
    'numpy'
]

setup(
    name='lewd-anime-classifier',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras Anime Classifier'
)
