tmux kill-session -t bearnav

tmux new-session -d -s "bearnav" -n "bearnav"
tmux new-window -d -n "mapmaker"
tmux new-window -d -n "repeater"
tmux new-window -d -n "misc"
tmux new-window -d -n "maps"
tmux new-window -d -n "resize"
tmux new-window -d -n "viz"

x=$(echo $SHELL | sed 's:.*/::')

tmux send-keys -t bearnav:bearnav "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:bearnav "roslaunch vtg jackal.launch "
tmux send-keys -t bearnav:mapmaker "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:mapmaker "rostopic pub /vtg/mapmaker/goal "
tmux send-keys -t bearnav:repeater "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:repeater "rostopic pub /vtg/repeater/goal "
tmux send-keys -t bearnav:misc "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /camera_front/image_color" Enter
sleep 3
tmux send-keys -t bearnav:misc "timeout 3 rostopic hz /husky_velocity_controller/odom" Enter
tmux send-keys -t bearnav:maps "cd ~/.ros" Enter
tmux send-keys -t bearnav:resize "rosrun nodelet nodelet standalone image_proc/resize image:=/camera2/color/image_raw camera_info:=/camera2/color/camera_info _scale_width:=0.8 _scale_height:=0.8"
tmux send-keys -t bearnav:viz "source ../../devel/setup.$x" Enter
tmux send-keys -t bearnav:viz "python ./gui/src/gui/particles-viz.py"
