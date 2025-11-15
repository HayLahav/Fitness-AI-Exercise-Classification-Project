"""
Setup configuration for Fitness AI Exercise Classification System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fitness-ai-exercise-classifier",
    version="1.0.0",
    author="Fitness AI Team",
    description="AI-powered exercise classification using computer vision and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Fitness-AI-Exercise-Classification-Project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.2",
            "pytest-cov>=5.0.0",
            "black>=24.8.0",
            "flake8>=7.1.1",
            "pylint>=3.2.7",
            "mypy>=1.11.2",
        ],
        "docs": [
            "mkdocs>=1.6.1",
            "mkdocs-material>=9.5.34",
        ],
    },
    entry_points={
        "console_scripts": [
            "fitness-ai-api=fitness_ai.api.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="exercise classification computer-vision deep-learning pose-estimation mediapipe",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Fitness-AI-Exercise-Classification-Project/issues",
        "Source": "https://github.com/yourusername/Fitness-AI-Exercise-Classification-Project",
    },
)
