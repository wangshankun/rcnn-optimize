time ./tools/train_net.py  \
     --gpu 0 \
     --weights output/rfcn_end2end_ohem/train/gs_plate_roi_local_float_iter_100000.caffemodel \
     --solver model/gs_plate_roi/ZF512/rfcn_end2end_relu_bn_good/solver_ohem_global.prototxt \
     --imdb gs_plate_roi_train \
     --iters 100000  \
     --cfg experiments/cfgs/gs_plate_roi_rfcn_zf.yml \
