# Enhanced-Path-Planning-through-Comparative-Analysis-of-RRT-and-Its-Variants-on-TurtleBot-in-Gazebo

## INSTRUCTIONS TO RUN

Download/Clone the package into the workspace and build using catkin build or catkin_make.
Source the workspace.

## 2-D Path Planning Methods

To run the below path planning algorithms, run the following from src folder the workspace:

### RRT 
```bash
$ cd final_project/scripts/
$ python3 rrt.py
```

### RRT*
```bash
$ cd final_project/scripts/
$ python3 rrt_star.py
```

### RRT*-Smart
```bash
$ cd final_project/scripts/
$ python3 rrt_star_smart.py
```

### RRT-Connect
```bash
$ cd final_project/scripts/
$ python3 rrt_connect.py
```


## Gazebo+Rviz Simulation
This is using Navigation stack so is technically not correct:
However depending on the algorithm to run uncomment the lines in the move.py script 
For RRT star, RRT star smart uncomment line 78 and comment line 79 for RRT connect and RRT do visa versa.

To run the simulation:

```bash
$ roslaunch final_project world.launch
```

In another terminal run,
```bash
$  roslaunch final_project navigation.launch 
```

Make the map align with the world in rviz by moving the robot using 2D Pose Estimate.

Open a new terminal and run,
```bash
$ cd final_project/scripts/
$ python3 move.py
```

Results:

RRT Execution:
https://youtu.be/I-2ZhwwAZuY
RRT Connect Execution:
https://youtu.be/5o3HtRhUp2k
RRT Star Execution:
https://youtu.be/Em3HYEddEJs
RRT Start Smart Execution:
[![Video Placeholder](https://img.youtube.com/vi/HC70_QCKaj4/0.jpg)](https://www.youtube.com/watch?v=HC70_QCKaj4)

2D - Results:

![Screenshot (34)](https://github.com/nishantpandey4/RRT-and-its-types/assets/127569735/506f9a5f-d13f-40cb-93d5-c32b0de7e5b4)


![Screenshot (35)](https://github.com/nishantpandey4/RRT-and-its-types/assets/127569735/da973bf0-01c8-4e20-b95e-d2af4d6f8b91)

