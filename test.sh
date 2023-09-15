python inference.py -a resnet18 \
--pretrained weights/baseline_gam/pixel_finetuning/checkpoint.pth.tar \
--data_path /data/comptition/jittor3/Jittor2_test_b \
--dump_path ./results \
-c 50 \
--mode test \
--match_file weights/baseline_gam/pixel_finetuning/validation/match.json