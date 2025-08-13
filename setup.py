from setuptools import setup, find_packages

setup(
    name="exoplanets-llopis-mary",
    version="0.1.0",
    description="Projet d'analyse de données exoplanétaires avec ML",
    author="Llopis / Mary",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "pynvml",
        "matplotlib",
        "rich",
        "ipykernel",
        "python-dotenv",
        "astropy",
        "emcee",
        "corner",
        "tqdm",
    ],
)
