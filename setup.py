from setuptools import setup, find_packages

setup(
    name='variant-classifier-md',
    version='0.0.1',
    description='Variant Classification using MD Simulations',
    license='BSD 3-clause license',
    maintainer='Chon Lok Lei',
    maintainer_email='chonloklei@um.edu.mo',
    packages=find_packages(include=('method')),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'joblib',
        'matplotlib',
        'scikit-learn',
        'tensorflow==2.4.0',
        'keras-tuner',
        'seaborn',
        'imbalanced-learn',
    ],
    extras_require={
        'jupyter': ['jupyter'],
    },
)
