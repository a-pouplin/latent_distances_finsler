from setuptools import find_packages, setup

"""
pip install -e .
"""

setup_requires = ["pytest-runner"]
tests_require = ["pytest"]

install_requires = ["numpy>=1.20.0", "torch>=1.9.0", "scikit-learn>=1.0.2"]


setup(
    name="finsler",
    version="1.0.0",
    description="Identifying latent distances with Finslerian geomtry",
    author="anonympus",
    packages=find_packages(),
    setup_requires=setup_requires,
    tests_require=tests_require,
    install_requires=install_requires,
    python_requires=">=3.8",
)
