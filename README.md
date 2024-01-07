# Enhanced-Path-Planning-through-Comparative-Analysis-of-RRT-and-Its-Variants-on-TurtleBot-in-Gazebo

This repository showcases enhanced path planning through a comparative analysis of RRT and its variants, including RRT, RRT*, RRT*-Smart, and RRT-Connect, implemented on a TurtleBot in the Gazebo simulation environment.

## Overview

This project focuses on optimizing path planning algorithms to achieve efficient navigation for TurtleBot in both 2D and 3D environments. The implemented RRT variants are evaluated, and their performances are compared, providing valuable insights into their strengths and weaknesses.

## 2D Simulaton Results:

  ![2D Results 1](https://github.com/nishantpandey4/RRT-and-its-types/assets/127569735/506f9a5f-d13f-40cb-93d5-c32b0de7e5b4)

  ![2D Results 2](https://github.com/nishantpandey4/RRT-and-its-types/assets/127569735/da973bf0-01c8-4e20-b95e-d2af4d6f8b91)


## 3D Simulation Results in Gazebo on TurtleBot (Youtube Video links below)
### RRT Execution:
[![RRT Execution](https://img.youtube.com/vi/I-2ZhwwAZuY/0.jpg)](https://www.youtube.com/watch?v=I-2ZhwwAZuY)

### RRT-Connect Execution:
[![RRT-Connect Execution](https://img.youtube.com/vi/5o3HtRhUp2k/0.jpg)](https://www.youtube.com/watch?v=5o3HtRhUp2k)

### RRT* Execution:
[![RRT* Execution](https://img.youtube.com/vi/Em3HYEddEJs/0.jpg)](https://www.youtube.com/watch?v=Em3HYEddEJs)

### RRT*-Smart Execution:
[![RRT*-Smart Execution](https://img.youtube.com/vi/HC70_QCKaj4/0.jpg)](https://www.youtube.com/watch?v=HC70_QCKaj4)

## Instructions to Run

1. **Setup:**
   - Download or clone the repository.
   - Build the workspace using `catkin build` or `catkin_make`.
   - Source the workspace.

2. **Run 2-D Path Planning Algorithms:**
   Navigate to the `final_project/scripts/` folder and execute the desired algorithm.

   - RRT 
     ```bash
     $ cd final_project/scripts/
     $ python3 rrt.py
     ```

   - RRT*
     ```bash
     $ cd final_project/scripts/
     $ python3 rrt_star.py
     ```

   - RRT*-Smart
     ```bash
     $ cd final_project/scripts/
     $ python3 rrt_star_smart.py
     ```

   - RRT-Connect
     ```bash
     $ cd final_project/scripts/
     $ python3 rrt_connect.py
     ```

3. **Run Gazebo + Rviz Simulation:**
   - Uncomment relevant lines in the `move.py` script based on the algorithm choice.
   - Run the simulation:

     ```bash
     $ roslaunch final_project world.launch
     ```

   - In another terminal, launch the navigation stack:

     ```bash
     $ roslaunch final_project navigation.launch 
     ```

   - Adjust the map alignment in RViz using 2D Pose Estimate.
   - Open a new terminal and execute:

     ```bash
     $ cd final_project/scripts/
     $ python3 move.py
     ```

## References:
For an additional implementation and insights, you can refer [here](https://github.com/anikk94/enpm661_project5/tree/main/final_project).
