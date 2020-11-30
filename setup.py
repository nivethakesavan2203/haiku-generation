"""

Setup for haiku_generation package.

"""
import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf8', 'ignore')

setup(
    name='haiku_generation',
    version='0.0.1',
    description='Haiku Generation Python package',
    long_description=readme,
    author='Nivetha Kesavan',
    author_email='nivetha.kesavan@colorado.edu',
    url='https://github.com/nivethakesavan2203/haiku-generation',
        install_requires=read_requirements(),
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
