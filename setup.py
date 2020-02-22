from setuptools import setup
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='RL',
      description="Some algorithms for solving RL problems",
      author="Abhinav Bhatia",
      url='https://github.com/bhatiaabhinav/RL',
      author_email="bhatiaabhinav93@gmail.com",
      version="1.0")
