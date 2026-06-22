from glob import glob

from setuptools import find_packages, setup

package_name = 'Posture_and_Gesture'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gw',
    maintainer_email='adgjl06@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gesture_detect_node = Posture_and_Gesture.gesture_detect:main',
            'posture_detect_node = Posture_and_Gesture.posture_detect:main',
            'action_recorder_node = Posture_and_Gesture.action_recorder:main',
        ],
    },
)
