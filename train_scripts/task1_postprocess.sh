cd /usr/src/MOD-CL/
python main.py 1 /usr/src/road-dataset/ ./ ./ --MODE=train --YOLO --MODE eval_frames --YOLO_PRETRAINED $1
cd ./result_output/