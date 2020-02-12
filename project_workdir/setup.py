# -*- coding: utf-8 -*-
# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='InvestiGame',
      version='0.0.1',
      description='Classify and predict performance of video games',
      author='Ashwin Rai',
      email='ashwin2rai@gmail.com',
      packages=['no_more_backlogs'],
	install_requires=['matplotlib','numpy==1.15.4','pycodestyle>=2.4.0'])


# Create proper setup to be used by pip
setup(name='AutoClassifier',
      version='0.0.1',
      description='Automatically performs binary classification',
      author='Ashwin Rai',
      email='ashwin2rai@gmail.com',
      packages=['autoclassifier'],
	install_requires=['matplotlib','numpy==1.15.4','pycodestyle>=2.4.0'])

