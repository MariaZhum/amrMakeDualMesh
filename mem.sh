
a=0                                                                                          
while true
do                                                                                   
	b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')    
	[ $b -gt $a ] && a=$b && echo $a
	sleep .001
done

