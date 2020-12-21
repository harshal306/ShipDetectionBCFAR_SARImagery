import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Detection-Algorithms-for-Ships-IIRS",
    version="0.0.1",
    author="Harshal Mittal",
    author_email="hrshlmittal306@gmail.com",
    description="This is small package that contains different algorithms for the detection of ships",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshal306/ShipDetectionBCFAR_SARImagery",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
   'numpy',
   'gdal',
   'easygui',
   'matplotlib',
   'KDEpy',
   'tqdm'
    ],   
)