from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT = '-e.'
def getrequirments(file_path:str)->List[str] :
    '''
    This function will return the list of the requirments
    '''
    requirments=[]
    with open (file_path) as file_obj :
        requirments = file_obj.readlines()
        requirments = [req.replace("\n","") for req in requirments]
        # if HYPEN_E_DOT in requirments :
        #     requirments.append(HYPEN_E_DOT)
        return requirments
setup(
    name = 'mlproject2',
    version='0.0.1',
    author='Ayushman Gupta',
    author_email='guptaayushman24@gmail.com',
    packages = find_packages(),
    install_requires = getrequirments('requirments.txt')
)
