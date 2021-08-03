#######################################################################
# Bash Script for inference and test speed
# --data_view choose from [updown, frontback, leftright]              #
#######################################################################


CUDA_VISIBLE_DEVICES=1 python3 inference_multi.py --data_view updown --prun_config_file config_ResNetUNet_0.5 --ext _tmp4_0.5 | tee out/out_updown_05.txt
CUDA_VISIBLE_DEVICES=2 python3 inference_multi.py --data_view updown --prun_config_file config_ResNetUNet_0.9 --ext _tmp3 | tee out/out_updown_09.txt
CUDA_VISIBLE_DEVICES=3 python3 inference_multi.py --data_view updown --prun_config_file config_ResNetUNet_0.99 --ext _tmp5_0.99 | tee out/out_updown_099.txt