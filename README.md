# irld_solvers

git clone https://github.com/dcarp/cmake-d.git
cd cmake-d
mkdir build
cd build
cmake ../cmake-d
sudo make install

sudo apt-get gdc ("D compiler (language version 2), based on the GCC backend")
  worked  
delete compiler other than gdc. sudo apt-get purge dmd. 
  

In a new terminal:

sudo rosdep init
rosdep update
cd ~/ros_workspace
catkin_make
