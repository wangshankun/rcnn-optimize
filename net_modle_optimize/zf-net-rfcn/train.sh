#!/bin/sh
python ./tools/train_net.py --gpu 0 --solver rt3399/data/roi/dw/solver_roi.prototxt --imdb gs_plate_roi_train --iters 300000 --cfg rt3399/data/roi/dw/roi.yml 

