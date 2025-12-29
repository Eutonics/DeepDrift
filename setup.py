from setuptools import setup, find_packages

setup(
    name="deepdrift",
    version="0.1.0",
    description="A Layer-Wise Diagnostic Framework for Neural Network Robustness",
    author="Alexey Evtushenko",
    author_email="alexey@deepdrift.ai",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy",
        "tqdm",
        "matplotlib",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
