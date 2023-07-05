from setuptools import find_packages, setup
from typing import List

HYPEN_DOT = '-e .'

def get_requirements(filepath:str)->List[str]:
    '''
    this function will return list of requirements
    '''

    requirements=[]
    with open(filepath) as fileobj:
        requirements = fileobj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

    if HYPEN_DOT in requirements:
        requirements.remove(HYPEN_DOT)

    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Ketan pande',
    author_email='ketanpande309@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)