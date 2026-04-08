from setuptools import setup


package_name = 'hri_voice_face_matcher'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hri_voice_face_matcher.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AGI',
    maintainer_email='agi@example.com',
    description='Voice to face/person association for ROS4HRI',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hri_voice_face_matcher = hri_voice_face_matcher.node_voice_face_matcher:main',
        ],
    },
)
