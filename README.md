# daVinciToolTrack
A vision-kinematic fusion approach for articulated tool tracking. Example video results can be found from https://youtu.be/oqw_9Xp_qsw.

If you use the code, please cite the following paper:

https://arxiv.org/pdf/1605.03483.pdf

@inproceedings{ye2016real,

  title={Real-time 3D Tracking of Articulated Tools for Robotic Surgery},
  
  author={Ye, Menglong and Zhang, Lin and Giannarou, Stamatia and Yang, Guang-Zhong},
  
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  
  pages={386--394},
  
  year={2016},
  
  organization={Springer}
  
 }
 

### Prerequisites ###

QT 5.5+

OpenCV 2.49+.

VTK 6.30+

Eigen (included)

Note that you will need to contact Intuitive Surgical for the CAD models of the surgical tools. Unfortunately, we are not allowed to share the CAD files without permission. 


### License ###

This code is distributed under BSD License.

### Notes ###

1. The provided implementation tracks up to two articulated da Vinci tools. 

2. A modern CPU processor should enable real-time processing speed of this implementation.

3. You can comment "#define MULTITHREADING" in "simulator.cpp" if you don't need parallel simulation and processing.
 
4. Although a running example is given for offline processing, it is straightforward to combine it with kinematics retrieved from dVRK or Intuitive API.

5. This implementation has been tested both in Windows and Ubuntu.

6. Further details will be provided for configuration files.
