from setuptools import setup, find_packages

package_name = 'qcar2_object_detections'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='QCar2 Developer',
    maintainer_email='developer@qcar2.local',
    description='QCar2 Object Detection package for Isaac ROS 2.1 YoloV8',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_preprocessor_node = qcar2_object_detections.image_preprocessor_node:main',
            'detection_filter_node = qcar2_object_detections.detection_filter_node:main',
            'detection_visualizer_node = qcar2_object_detections.detection_visualizer_node:main',
        ],
    },
)
