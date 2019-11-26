for w in 640 768 896
do
for s in 1 2
do
    echo "$w"_$(($w*$s))
    python map_FE.py 1 $w $(($w*$s))
    python map_FE.py 2 $w $(($w*$s))
    python map_FE.py 3 $w $(($w*$s))
done
done