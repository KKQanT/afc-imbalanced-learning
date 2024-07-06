from setuptools import setup, find_packages

setup(
    name='afc_svm_imbalanced_learning',
    version='0.5.0',
    author='Peerakarn',
    author_email='peerakarn.jit@gmail.com',
    description='Simple implementation of the papar Adaptive Feature-Space Conformal Transformation for Imbalanced-Data Learning',
    long_description=open('README.md', encoding="utf8").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KKQanT/afc-imbalanced-learning',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        'dev': [
            'pandas'
            'pytest',
            'unittest',
            'ucimlrepo'
        ],
    },
)
