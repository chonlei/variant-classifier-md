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
        'numpy==1.19.5',
        'scipy==1.6.0',
        'pandas==1.2.1',
        'joblib==1.0.0',
        'matplotlib==3.3.3',
        #'scikit-learn==0.24.0',
        'scikit-learn==1.0.0',
        'tensorflow==2.4.0',
        'tensorflow-addons==0.13.0',
        'keras-tuner==1.0.2',
        'imbalanced-learn==0.8.1',
        'seaborn==0.11.1',
    ],
    extras_require={
        'jupyter': ['jupyter'],
    },
)
