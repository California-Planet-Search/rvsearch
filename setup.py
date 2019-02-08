from setuptools import setup, find_packages
import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

reqs = []
for line in open('requirements.txt', 'r').readlines():
    reqs.append(line)

setup(
    name="rvsearch",
    version=get_property('__version__', 'rvsearch'),
    author="Lee Rosenthal, BJ Fulton",
    packages=find_packages(),
    entry_points={'console_scripts': ['rvsearch=rvsearch.cli:main']},
    install_requires=reqs
)
