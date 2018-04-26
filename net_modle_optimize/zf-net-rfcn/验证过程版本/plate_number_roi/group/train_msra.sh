time ./tools/train_net.py  \
     --gpu 0 \
     --solver rfcn/group/solver_msra.prototxt \
     --imdb gs_plate_roi_train \
     --iters 100000  \
     --cfg rfcn/group/zf.yml


