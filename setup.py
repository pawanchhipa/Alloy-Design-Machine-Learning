from setuptools import setup, find_packages

setup(
    name="alloy-ml-prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.23.0'
    ],
    author="Pawan Chhipa",
    author_email="pawanchhipa50@gmail.com",
    description="A package for alloy design and prediction using machine learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pawanchhipa/Alloy-Design-Machine-Learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Materials Science"
    ],
    python_requires='>=3.6',
)
