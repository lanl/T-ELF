from setuptools import setup, find_packages
from glob import glob
__version__ = "0.0.4"

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='TELF',
    version=__version__,
    author='',
    author_email='',
    description='Tensor Extraction of Latent Features (TELF)',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    package_dir={'TELF': 'TELF'},
    platforms = ["Linux", "Mac", "Windows"],
    include_package_data=True,
    setup_requires=INSTALL_REQUIRES,
    url='https://github.com/lanl/T-ELF',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.11.5',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.11.5',
    install_requires=INSTALL_REQUIRES,
    license='License :: BSD License',
)
