import argparse
import subprocess
cmd_str = (f'python inference.py '
           f'-a resnet18 '
           f'--pretrained weights/checkpoint.pth.tar '
           f'--data_path /data/comptition/jittor3/Jittor2_test_b '
           f'--dump_path ./results '
           f'-c 50 '
           f'--mode test '
           f'--match_file weights/match.json ')
subprocess.call(cmd_str, shell=True)
