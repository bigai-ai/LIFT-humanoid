"""setup.py for LIFT project.

Install for development:
  pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="lift",
    version="0.1.0",
    description="LIFT: Physics Informed Model-Based Policy Optimization",
    long_description_content_type="text/markdown",
    url="https://lift-humanoid.github.io/",
    license="Apache 2.0",
    packages=find_packages(exclude=["tests", "tests.*", "wandb", "outputs", "logs", "models"]),
    include_package_data=True,
    install_requires=[
        "absl-py",
        "flax",
        "jax",
        "jaxlib",
        "numpy<2.0",
        "optax",
        "scipy",
        "tensorboardX",
        "wandb",
        "dill",
        "ml_collections",
        "omegaconf",
        "hydra-core",
        "jaxopt==0.8.3",
    ],
    extras_require={
        "develop": ["pytest", "pytest-xdist"],
    },
    python_requires=">=3.8",
)