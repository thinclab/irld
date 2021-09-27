# irld_solvers

git clone https://github.com/dcarp/cmake-d.git
cd cmake-d
mkdir build
cd build
cmake ../cmake-d
sudo make install

In a new terminal:

sudo rosdep init
rosdep update
cd ~/ros_workspace
catkin_make
