import os, sys
from setuptools import setup, find_packages
#from PEPPA import __VERSION__
__VERSION__ = '0.1'

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='fastSource',
    version= __VERSION__,
    #scripts=['PEPPA.py'] ,
    author="Zhemin Zhou",
    author_email="zhemin.zhou@warwick.ac.uk",
    description="Automatic classification of Source Metadata.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zheminzhou/fastSource",
    packages = ['fastSource'],
    package_dir = {'fastSource':'.'},
    keywords=['bioinformatics', 'microbial', 'genomics', 'metadata', 'machine learning', 'text classification'],
    install_requires=['Flask>=1.1.2', 'gunicorn>=20.0.4', 'numpy>=1.18.1', 'Flask-Cors>=3.0.8', 'fasttext>=0.9.2', 'Click>=7.0', 'ujson>=2.0.3', 'numpy>=1.18.3', 'pandas>=1.0.3'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'fastSource = fastSource.fastSource:main',
            'fastSource_build = fastSource.fastSource_build:main',
    ]},
    package_data={'fastSource': ['LICENSE', 'README.*']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
 )

