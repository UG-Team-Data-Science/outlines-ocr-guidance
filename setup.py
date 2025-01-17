# prompt: create files for a pip pypi package called boxly, a shapely alternative for bounding boxes. Also create a readme. Prefix each file with a comment line of the filename, create at least a setup.py and a README.md

# setup.py
from setuptools import setup, find_packages

setup(
    name='outlines_ocr_guidance',
    version='0.1.0',
    packages=find_packages(),
    install_requires=["outlines>=0.0.46", ],
    author='H.T. Kruitbosch',
    author_email='outlines-ocr-guidance@herbertkruitbosch.com',
    description="outlines LLM guide to return YAML-structured spans from texts, typically OCRs from old registries.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/prhbrt/outlines-ocr-guidance/',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)