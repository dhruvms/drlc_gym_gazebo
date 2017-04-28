import numpy as np
import os

this_script_dir = os.path.dirname(os.path.realpath(__file__))
print this_script_dir

# setup file to be written
filename='/home/vaibhav/madratman/projects/inspection_sim_ws/src/inspection_sim/dji_gazebo/worlds/final_forest_final.world'
target=open(filename,'w')

# read beginning of file
string = open(os.path.join(this_script_dir, 'begin3_madratman'), 'r').read()
cil_1 = open(os.path.join(this_script_dir, 'cil_13_madratman'), 'r').read()
cil_2 = open(os.path.join(this_script_dir, 'cil_23_madratman'), 'r').read()

# write beginning of file
target.write(string)

# generate random samples
nx = 15
spacing_x = 6
random_interval_x = spacing_x/3
offset_x = 5

ny = 10
spacing_y = 6
random_interval_y = spacing_y
offset_y = -int(ny*spacing_y/2)+3

x = np.linspace(offset_x, offset_x+(nx-1)*spacing_x, nx)
y = np.linspace(offset_y, offset_y+(ny-1)*spacing_y, ny)

positions_x=np.zeros([nx,ny])
positions_y=np.zeros([nx,ny])

counter=0
np.random.seed() #use seed from sys time to build new env on reset
for i in range(nx):
	for j in range(ny):
		name="\n    <model name='unit_cylinder_"+str(counter)+"'>"
		counter+=1
		target.write(name)
		noise_x=np.random.random()-0.5
		noise_x*=random_interval_x
		noise_y=np.random.random()-0.5
		noise_y*=random_interval_y
		x_tree=x[i]+noise_x
		y_tree=y[j]+noise_y
		positions_x[i,j]=x_tree
		positions_y[i,j]=y_tree
		line_to_print="\n      <pose frame=''>"+str(x_tree)+" "+str(y_tree)+" 5 0 -0 0</pose>\n"
		target.write(line_to_print)
		target.write(cil_1)

target.write("\n")
target.write( "</world>\n</sdf>\n")
target.close()

np.savetxt('pos_x.out', positions_x, delimiter=',')
np.savetxt('pos_y.out', positions_y, delimiter=',')

