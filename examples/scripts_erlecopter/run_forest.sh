kill -9 `ps aux | grep ros | awk '{print $2}'`
kill -9 `ps aux | grep gazebo | awk '{print $2}'`
python navigate_erlecopter_dqn.py