# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

package_name = 'asr_vosk'


setup(
    name=package_name,
    version='2.5.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module', ['module/' + package_name]),
        ('share/ament_index/resource_index/pal_configuration.' + package_name,
            ['config/' + package_name]),
        ('share/ament_index/resource_index/i18n.localized_nodes',
            ['localized_nodes/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/00-defaults.yml']),
        ('share/' + package_name + '/config', ['config/asr_vosk_analyzers.yaml']),
        ('share/' + package_name + '/launch', ['launch/asr_vosk.launch.py']),
        ('share/' + package_name + '/launch', ['launch/asr_vosk_with_mic_and_matcher.launch.py']),
        ('share/' + package_name + '/module', ['module/asr_vosk_module.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Séverin Lemaignan',
    maintainer_email='severin.lemaignan@pal-robotics.com',
    description='The asr_vosk package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'asr_vosk = asr_vosk.node_vosk:main',
                'pal_asr_vosk = asr_vosk.pal_node_vosk:main',
                'alsa_mic = asr_vosk.mic:main',
        ],
    },
)
