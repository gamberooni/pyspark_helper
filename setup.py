import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pyspark_helper",
    version = "0.1.0",
    author = "Zhaoyong Tan",
    author_email = "zhaoyoong@gmail.com",
    description = ("Pyspark helper functions"),
    license = "BSD",
    keywords = "pyspark helper",
    url = "https://github.com/gamberooni/pyspark_helper",
    packages=['pyspark_helper'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)