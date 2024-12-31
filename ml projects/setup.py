from typing import List
from setuptools import find_packages,setup



def get_requirements(file_path)->  List[str]:

    '''
        this function will return the list of requirements
    '''
    requirements=[]

    with open(file_path) as file_ob:
        requirements=file_ob.readlines()
        requirements=[a.replace("\n","") for a in requirements]
    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements




setup(
    name="mlproject",
    version='0.0.1',
    author="janaki ram",
    author_email="prodduturujanakiram@gmail.com"  ,
    packages=find_packages() ,
    install_requires=get_requirements("requirements.txt")
)