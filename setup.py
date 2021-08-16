import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["abcd*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="abcd",
    version="0.0.1",
    author="Derek Chen",
    author_email="dchen@asapp.com",
    description="pip package for the ABCD repo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asappresearch/abcd",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=["torch", "transformers", "python-dateutil", "simple_parsing"],
)
