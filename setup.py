from setuptools import setup, find_packages

setup(name='pybbfmm',
      version='0.1',
      description='A black-box fast multipole method built on top of PyTorch',
      author='Andy Jones',
      author_email='andyjones.ed@gmail.com',
      url='https://github.com/andyljones/pybbfmm',
      packages=find_packages(),
      python_requires='>=3.6',
      install_requires=[
          'torch>=1.4', 'aljpy>=0.4', 'numpy>=1.18'],
      extra_requires={'demo': [
          'matplotlib>=3.2', 'scipy>=1.4', 'ipython>=7.5', 'tqdm>=4.42', 'requests>=2.22']}) 

