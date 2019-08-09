from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'inflection==0.3.1',
    'matplotlib==3.1.1',
    'python-dateutil==2.8.0',
    'requests==2.22.0',
    'tensorflow==1.14.0',
    'scikit-learn==0.21.3',
    'numpy==1.17.0'
]

setup(
    name='anime_classifier',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras Anime Classifier'
)
