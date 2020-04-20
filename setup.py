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
      extras_require={'demo': [
          # av~=6.2 as later versions require ffmpeg 4, which isn't in most distro's repos.
          'matplotlib>=3', 'scipy>=1.4', 'ipython>=5', 'tqdm>=4', 'requests>=2', 'av~=6.2']}) 

