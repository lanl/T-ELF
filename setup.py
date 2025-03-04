from setuptools import setup, find_packages
from glob import glob
__version__ = "0.0.36"

setup(
    name='TELF',
    version=__version__,
    author='',
    author_email='',
    description='Tensor Extraction of Latent Features (TELF)',
    long_description_content_type='text/markdown',
    package_dir={'TELF': 'TELF'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    url='https://github.com/lanl/T-ELF',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.11.10',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.11.10',
    license='License :: BSD License',
)