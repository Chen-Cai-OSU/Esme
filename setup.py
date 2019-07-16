import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Esme",
    version="0.0.1",
    author="Chen Cai",
    author_email="cai.507@osu.edu",
    description="Library for topological data analysis of graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=[
        "networkx",
        "dionysus",
        "sklearn",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
