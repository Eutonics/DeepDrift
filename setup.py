from setuptools import setup, find_packages

setup(
    name="deepdrift",
    version="1.0.0",
    description="Universal Thermodynamic Framework for Neural Network Robustness (Vision, LLM, Router)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexey Evtushenko",
    author_email="alexey@eutonics.ru",
    url="https://github.com/Eutonics/DeepDrift",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
    ],
    keywords="llm hallucination detection, vision robustness, ood detection, semantic velocity"
)
