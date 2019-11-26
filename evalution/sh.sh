export DISPLAY=:0.0
for w in "640 640" "640 1280" "768 768" "896 896"
do
for b in 1 2
do
	echo $w $b
	sleep 10
	python fishEye_lib.py  --size $w --batch $b	
done
done
test_model_640_1280.onnx

root = "/home/user/run_retina/weights"
for w in 640 768 896
do
for s in 1 2
do
    echo "$w"_$(($w*2))
    ./export /home/user/project/run_retina/weights/test_model_"$w"_$(($w*2)).onnx /home/user/project/run_retina/weights/fcos_int8_"$w"_$(($w*2)).plan caliTable_int8_"$w"x$(($w*2))
done
done