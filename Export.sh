for w in 640 768 896
do
for s in 1 2
do
    echo "$w"_$(($w*$s))
    ./build/export /home/user/project/run_retina/weights/test_model_"$w"_$(($w*$s)).onnx /home/user/project/run_retina/weights/fcos_int8_"$w"_$(($w*$s)).plan /home/user/project/run_retina/weights/caliTable_int8_"$w"x$(($w*$s))
done
done