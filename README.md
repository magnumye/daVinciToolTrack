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

The model files can be downloaded from http://research.intusurg.com/dvrkwiki/index.php?title=DVRK:Topics:InstrumentModels. But you may need to contact Intuitive Surgical (research@intusurg.com) for access.


### License ###

This code is distributed under BSD License.

### Notes ###

1. The provided implementation tracks up to two articulated da Vinci tools. 

2. A modern CPU processor should enable real-time processing speed of this implementation.

3. You can comment "#define MULTITHREADING" in "simulator.cpp" if you don't need parallel simulation and processing.
 
4. Although a running example is given for offline processing, it is straightforward to combine it with kinematics retrieved from dVRK or Intuitive API for live tracking.

5. This implementation has been tested both in Windows and Ubuntu.

