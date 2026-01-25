from setuptools import setup, find_packages

# Читаем README для PyPI
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deepdrift",
    version="0.4.0",
    author="Alexey Evtushenko",
    author_email="alexey@eutonics.ru",
    description="A universal thermodynamic framework for neural network robustness monitoring.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eutonics/DeepDrift",
    project_urls={
        "Bug Tracker": "https://github.com/Eutonics/DeepDrift/issues",
        "Demo": "https://huggingface.co/spaces/Eutonics/DeepDrift-Explorer",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy",
        "scipy",
    ],
)
