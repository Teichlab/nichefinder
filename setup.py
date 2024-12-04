from setuptools import setup, find_packages

setup(
    name="iss2niche",
    version="0.1.0",
    description="A project for niche analysis using spatial and suspension datasets",
    author="J.P.Pett",
    author_email="jp30@sanger.ac.uk",
    url="https://github.com/Teichlab/snp2cell",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scanpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "joblib",
        "networkx",
        "numpy-groupies",
    ],
    extras_require={"dev": ["pytest", "flake8", "black"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5, <3.12",
)
