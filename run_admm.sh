#######################################################################
# Script for the model compression: ADMM train                        #
#######################################################################


export cuda_num=$1
# export view=$2
export prune_ratio=$2
export ext_name=$3

# view_lst=('updown' 'leftright' 'frontback')
# view_lst="updown leftright frontback"

view_lst="frontback"

for view in $view_lst
do
    echo "Start running $view"
    WANDB_RUN_ID=slr-$view-$prune_ratio-$ext_name CUDA_VISIBLE_DEVICES=$cuda_num python3 train_admm.py --admm_train\
                                                                                            --data_view $view\
                                                                                            --prun_config_file config_ResNetUNet_$prune_ratio\
                                                                                            --ext $prune_ratio\_$ext_name\
                                                                                            | tee output/out_admmtrain_$view\_$prune_ratio\_$ext_name.txt
    WANDB_RUN_ID=retrain-$view-$prune_ratio-$ext_name CUDA_VISIBLE_DEVICES=$cuda_num python3 train_admm.py --masked_retrain\
                                                                                                --data_view $view\
                                                                                                --prun_config_file config_ResNetUNet_$prune_ratio\
                                                                                                --ext $prune_ratio\_$ext_name\
                                                                                                | tee output/out_maskedretrain_$view\_$prune_ratio\_$ext_name.txt
done