from setuptools import setup, find_packages
setup(
    name='iterative_scVI_clustering',
    version='0.1.0',
    author='Michael J. Deines',
    author_email='michaeljdeines@gmail.com',
    description='A package for iterative clustering based on scVI.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mikejdeines/Iterative_scVI_Clustering',
    packages=find_packages(include=['iterative_scVI_clustering*']),
    install_requires=[
        'scanpy',
        'pandas',
        'igraph',
        'leidenalg',
        'scvi-tools>=0.18.1',
        'numpy>=1.24.4,<2.0.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: macOS or Linux',
    ],
    python_requires='>=3.6',
)