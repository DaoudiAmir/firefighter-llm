from setuptools import setup, find_packages

setup(
    name="firefighter_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "jupyter",
        "pytest",
        "beautifulsoup4",
        "requests",
        "PyMuPDF",
        "python-dotenv",
        "tqdm",
        "PyYAML"
    ],
)
