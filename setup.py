from setuptools import find_packages, setup


def get_requirements(path):
    requirements = []
    with open(path, 'r') as txt_file:
        requirements = txt_file.readlines()
        [req.replace("\n"," ") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')

        
    return requirements

setup(
    name='Image Captioning',
    version='0.0.1',
    author='Manas Sharma',
    author_email='manassharma.ms2593@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)