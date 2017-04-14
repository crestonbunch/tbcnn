"""Setup script to install tbcnn scripts."""

from setuptools import setup, find_packages

setup(
    name="tbcnn",
    version=0.1,
    description="Perform deep learning analysis on Python scripts",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "crawl=crawler.commands:main",
            "sample=sampler.commands:main",
            "vectorize=vectorizer.commands:main",
            "classify=classifier.commands:main",
        ]
    }
)