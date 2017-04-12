bash -c 'echo source `pwd`/devel/setup.bash >> ~/.bashrc'
echo "## ROS workspace compiled ##"

#add own models path to gazebo models path
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../../assets/models >> ~/.bashrc'
  exec bash #reload bashrc
fi

# Theano and Keras installation and requisites
cd ../
sudo pip install h5py
sudo apt-get install gfortran
git clone git://github.com/Theano/Theano.git
cd Theano/
sudo python setup.py develop
sudo pip install keras

