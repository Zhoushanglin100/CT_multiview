#######################################################################
# Bash Script for testing the speed                                   #
#######################################################################


export save_file=$1

for f in ./cvt_model/new_model/*.pt; do
    filename_full=${f##*/}
    echo $filename_full
    python3 test.py --load_model_name $filename_full >> output_info/$save_file
    echo >> output_info/$save_file
done
echo "Finish!!"