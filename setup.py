from setuptools import setup, find_packages

setup(
    name="deepdrift",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "tqdm",
        "scipy",
    ],
    extras_require={
        "all": [
            "transformers",
            "accelerate",
            "bitsandbytes",
            "gymnasium[box2d]",
            "stable-baselines3",
            "shimmy",
            "timm",
            "scikit-learn",
        ],
    },
    author="Alexey Evtushenko",
    description="Universal Kinetic Diagnosis for Neural Networks",
    url="https://github.com/Eutonics/DeepDrift",
)
