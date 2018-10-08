from setuptools import setup
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='RL',
      install_requires=[
          'gym[mujoco,atari,classic_control]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          'tensorflow',
          'keras',
          'azure==1.0.3',
          'progressbar2',
          'mpi4py',
          'matplotlib',
          'cplex'
      ],
      description="Some algorithms for solving RL algorithms",
      author="Abhinav Bhatia",
      url='https://github.com/bhatiaabhinav/RL',
      author_email="bhatiaabhinav93@gmail.com",
      version="1.0")
