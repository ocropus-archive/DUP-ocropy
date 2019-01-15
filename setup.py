from glob import glob
from setuptools import setup

models = [c for c in glob('models/*pyrnn.gz')]
scripts = [c for c in glob('ocropus-*') if '.' not in c and '~' not in c]

setup(
    name='ocropy',
    version='2.0.0a1',
    author='Thomas Breuel',
    maintainer='Konstantin Baierer',
    maintainer_email='unixprog@gmail.com',
    description='The OCRopy RNN-based Text Line Recognizer',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tmbdev/ocropy',
    license='Apache-2.0',
    python_requires='>=2.7',

    packages=['ocrolib'],
    data_files=[('share/ocropus', models)],
    scripts=scripts,
    include_package_data=True,

    install_requires=[
        'numpy      >= 1.15.4',
        'scipy      >= 1.1.0',
        'matplotlib >= 2.2.3',
        'imageio    >= 2.4.1',
        'Pillow     >= 2.7.0',
        'lxml       >= 3.5.0',
        'six        >= 1.10.0',
    ],
    keywords=['OCR', 'optical character recognition', 'ocropy', 'ocropus', 'kraken', 'calamari'],
)
