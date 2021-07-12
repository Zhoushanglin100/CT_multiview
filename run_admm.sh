#######################################################################
# Script for the model compression: ADMM train                        #
#######################################################################


export cuda_num=$1
export prune_ratio=$2
export ext_name=$3

WANDB_RUN_ID=admmtrain-$prune_ratio-$ext_name CUDA_VISIBLE_DEVICES=$cuda_num python3 main_admm.py --admm_train\
                                                                                          --prun_config_file config_ResNetUNet_$prune_ratio\
                                                                                          --ext $prune_ratio\_$ext_name\
                                                                                          | tee output/out_admmtrain_$prune_ratio\_$ext_name.txt
WANDB_RUN_ID=maskedretrain-$prune_ratio-$ext_name CUDA_VISIBLE_DEVICES=$cuda_num python3 main_admm.py --masked_retrain\
                                                                                              --prun_config_file config_ResNetUNet_$prune_ratio\
                                                                                              --ext $prune_ratio\_$ext_name\
                                                                                              | tee output/out_maskedretrain_$prune_ratio\_$ext_name.txt