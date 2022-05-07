import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mitar", # Replace with your own username
    version="0.0.1",
    author="Tongjun Gu",
    author_email="tgu@ufl.edu",
    description="miRNA target prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tjgu/miTAR",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
