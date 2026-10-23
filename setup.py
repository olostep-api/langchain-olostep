"""
Setup script for langchain-olostep package.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from pyproject.toml
requirements = [
    "langchain-core>=0.3.0",
    "requests>=2.31.0",
    "pydantic>=2.0.0",
]

setup(
    name="langchain-olostep",
    version="0.3.0",
    author="Olostep",
    author_email="info@olostep.com",
    description="The most reliable and cost-effective web search, scraping and crawling API for AI. Build intelligent agents that can search, scrape, analyze, and structure data from any website.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olostep-api/langchain-olostep",
    project_urls={
        "Bug Reports": "https://github.com/olostep-api/langchain-olostep/issues",
        "Source": "https://github.com/olostep-api/langchain-olostep",
        "Documentation": "https://docs.olostep.com/integrations/langchain",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "httpx>=0.24.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

