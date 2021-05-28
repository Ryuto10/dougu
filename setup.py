from setuptools import setup

setup(
    name='dougu',
    version='0.2',
    description='miscellaneous utilities',
    url='http://github.com/Ryuto10/dougu',
    author='Ryuto Konno',
    author_email='pxx.rai@gmail.com',
    license='MIT',
    packages=['dougu'],
    zip_safe=False, install_requires=['numpy', 'logzero', 'torch', 'torchtext', 'transformers', 'google']
)
