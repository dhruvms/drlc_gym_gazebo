kill -9 `ps aux | grep ros | awk '{print $2}'`
kill -9 `ps aux | grep gazebo | awk '{print $2}'`
python erlecopter_hover_qlearn.py
