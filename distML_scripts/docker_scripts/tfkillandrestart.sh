#!/bin/bash

#just to create separate log directory so output from previous run doesn't need to be saved elsewhere when running new experimental config.
#this script just assumes 1 master for now (w.r.t sleep and detach mode)
run_number="1/"
log_dir="/home/styagi/"
model_dir="model_dir"
# false is ASP, while true runs with BSP.
sync_mode="false"
trainsteps=1000
step_size=1000
logdirectory="/home/styagi/1/*/*.txt"
#make a directory to store logs for a given run
if [ ! -d $log_dir ]
then
        mkdir $log_dir
fi

if [ ! -d $log_dir$run_number ]
then
	mkdir $log_dir$run_number
fi

if [ ! -d $log_dir$run_number$model_dir ]
then
	mkdir $log_dir$run_number$model_dir
fi

job_dir=$log_dir$run_number$model_dir

#num_nodes is equal to the length of iplist. change this to set as external args and dynamically allocate these IPs based on # of ps, master and workers.
iplist=("10.18.0.7" "10.18.0.4" "10.18.0.5" "10.18.0.6")
stepsequence=("1000" "3000" "6000" "10000" "15000" "21000" "28000" "36000" "45000" "55000")
#change this for cpu allocations in heterogenous mode
cpucoresalloc=("9" "12" "9" "18")
memalloc=("16G" "60G" "30G" "36G")
batchsizes=("44" "59" "44" "89")
tcp_port=":8000"
num_ps=1
num_master=1
num_workers=2
num_nodes=$(($num_ps+$num_master+$num_workers))
network="tfnet"
tf_param="{\"environment\": \"cloud\", \"model_dir\": \"/root\", \"batch_size_list\": [44,59,44,89], \"cluster\": {"
reruns=9

# stop and remove previous containers
for i in $(seq 1 $num_nodes)
do
	tf_config=""
        service_id=""
        if [ $(($i-1)) -lt $num_ps ]
        then
                service_id="tf-ps-"$(($i-1)) 
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
		service_id="tf-master-"$(($i-$(($num_ps+1))))
        else
                service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	#if [ docker container ps -a | grep '$service_id' ]
	#then
		echo "going to stop and remove container "$service_id
        	docker container stop $service_id
        	docker container rm $service_id
	#fi
done

task_index=0
worker_str="\"worker\": ["
ps_str="\"ps\": ["
master_str="\"master\": ["
for i in $(seq 1 $num_nodes)
do
	if [ $(($i-1)) -eq 0 ]
        then
                #ps here
                tf_param+="\"ps\":[\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -lt $num_ps ]
	then
		#ps here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -eq $num_ps ]
	then
		tf_param+="],"
		#master here
                tf_param+="\"master\":[\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -gt $num_ps ] && [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
	then
		#master here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -eq $(($num_ps+$num_master)) ]
	then
		#1st worker string here
		tf_param+="],\"worker\":[\"${iplist[$(($i-1))]}$tcp_port\""
	else
		#worker here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	fi
done

tf_param+="]}, \"task\": "
#to add task type, index to tf_config and launch appropriate container with its corresponding cpu-cores allocation. \
#change this to dynamically pick task_cpus dynamically (another array or map) to accomodate heterogenous setups.
for i in $(seq 1 $num_nodes)
do
	tf_config=""
	service_id=""
	if [ $(($i-1)) -lt $num_ps ]
        then
		tf_config=$tf_param"{\"index\": $(($i-1)), \"type\": \"ps\"}}"
		service_id="tf-ps-"$(($i-1)) 
	elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
	then
		tf_config=$tf_param"{\"index\": $(($i-$(($num_ps+1)))), \"type\": \"master\"}}"
		service_id="tf-master-"$(($i-$(($num_ps+1))))
	else
		tf_config=$tf_param"{\"index\": $(($i-$(($(($num_ps+$num_master))+1)))), \"type\": \"worker\"}}"
		service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	#start containers
	echo "going to create container "$service_id
	echo "tf_config: "$tf_config
	echo "ip assigned: "${iplist[$(($i-1))]}
	echo "cpus alloted to task: "${cpucoresalloc[$(($i-1))]}
	echo "memory allocated to container: "${memalloc[$(($i-1))]}
	echo "model directory is $job_dir"
	docker run -it -w /resnet-cifar10 -v /home/styagi/prateeks/tensorflow:/resnet-cifar10 -v "$job_dir":/root \
	--env TF_CONFIG="$tf_config" --name $service_id --cpus ${cpucoresalloc[$(($i-1))]} -m ${memalloc[$(($i-1))]} --net $network --ip ${iplist[$(($i-1))]} \
	-d tflogitertime:v1 /bin/bash
done

docker stats &>> $log_dir$run_number/"dockerstats.txt" &
echo "start time logged in non-training loop "$start_time
./date.sh
#run resnet model code on the launched containers
for i in $(seq 1 $num_nodes)
do
        service_id=""
        if [ $(($i-1)) -lt $num_ps ]
 	then
                service_id="tf-ps-"$(($i-1)) 
        	echo "going to launch resnet model on container $service_id"
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
        	service_id="tf-master-"$(($i-$(($num_ps+1))))
        else
        	service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	if [ ! -d $log_dir$run_number$service_id ]
	then
		echo "directory to create:"$log_dir$run_number$service_id
		mkdir $log_dir$run_number$service_id
	fi

	echo "going to launch resnet model on container $service_id"
 	if [ "$sync_mode" == "false" ]
	then
		echo "running ASP mode"
	    docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py \
			--data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --num-gpus=0 \
			--train-steps='${stepsequence[0]}''  &>> $log_dir$run_number$service_id/"stepslogs.txt" &
	elif [ "$sync_mode" == "true" ]
	then
		echo "ALL EXCEPT LAST running in BSP mode as sync is set to true.."
		docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py \
        	--data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --num-gpus=0 \
            --train-steps='${stepsequence[0]}' --sync'  &>> $log_dir$run_number$service_id/"stepslogs.txt" &
	fi
done

#here a blocking call is made for string check
while true
do
	if [[ "$(grep -rn -i 'INFO:tensorflow:Loss for final step:' $logdirectory)" == *'INFO:tensorflow:Loss for final step:'* ]]
	then
		echo "going to break flow in run # 1....."
		break
	fi
done

echo "end time logged in non-training loop..."
./date.sh

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% going to start re-runs now.................."
for z in $(seq 1 $reruns)
do

	# make batch_size_list here for every time.

	# trainsteps=$((trainsteps + step_size))
	# echo "UPDATED TRAIN_STEPS VALUES IS:"
	# echo $trainsteps
	tf_param=""
	worker_str=""
	ps_str=""
	master_str=""
	tf_param="{\"environment\": \"cloud\", \"model_dir\": \"/root\", \"batch_size_list\": [64,49,10,133], \"cluster\": {"

	# stop and remove previous containers
	for i in $(seq 1 $num_nodes)
	do
		tf_config=""
        service_id=""
        if [ $(($i-1)) -lt $num_ps ]
        then
        	service_id="tf-ps-"$(($i-1)) 
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
			service_id="tf-master-"$(($i-$(($num_ps+1))))
        else
            service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
		fi

		echo "going to stop and remove container "$service_id
        docker container stop $service_id
        docker container rm $service_id
	done

	#create tfconfig and start containers again...
	task_index=0
	worker_str="\"worker\": ["
	ps_str="\"ps\": ["
	master_str="\"master\": ["
	for i in $(seq 1 $num_nodes)
	do
		if [ $(($i-1)) -eq 0 ]
        then
            #ps here
            tf_param+="\"ps\":[\"${iplist[$(($i-1))]}$tcp_port\""
		elif [ $(($i-1)) -lt $num_ps ]
		then
			#ps here
			tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
		elif [ $(($i-1)) -eq $num_ps ]
		then
			tf_param+="],"
			#master here
            tf_param+="\"master\":[\"${iplist[$(($i-1))]}$tcp_port\""
		elif [ $(($i-1)) -gt $num_ps ] && [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
		then
			#master here
			tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
		elif [ $(($i-1)) -eq $(($num_ps+$num_master)) ]
		then
			#1st worker string here
			tf_param+="],\"worker\":[\"${iplist[$(($i-1))]}$tcp_port\""
		else
			#worker here
			tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
		fi
	done

	tf_param+="]}, \"task\": "
	for i in $(seq 1 $num_nodes)
	do
		tf_config=""
		service_id=""
		if [ $(($i-1)) -lt $num_ps ]
        then
			tf_config=$tf_param"{\"index\": $(($i-1)), \"type\": \"ps\"}}"
			service_id="tf-ps-"$(($i-1)) 
		elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
		then
			tf_config=$tf_param"{\"index\": $(($i-$(($num_ps+1)))), \"type\": \"master\"}}"
			service_id="tf-master-"$(($i-$(($num_ps+1))))
		else
			tf_config=$tf_param"{\"index\": $(($i-$(($(($num_ps+$num_master))+1)))), \"type\": \"worker\"}}"
			service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
		fi

		#start containers
		echo "going to create container "$service_id
		echo "tf_config: "$tf_config
		echo "ip assigned: "${iplist[$(($i-1))]}
		echo "cpus alloted to task: "${cpucoresalloc[$(($i-1))]}
		echo "memory allocated to container: "${memalloc[$(($i-1))]}
		echo "model directory is $job_dir"
		docker run -it -w /resnet-cifar10 -v /home/styagi/prateeks/tensorflow:/resnet-cifar10 -v "$job_dir":/root \
		--env TF_CONFIG="$tf_config" --name $service_id --cpus ${cpucoresalloc[$(($i-1))]} -m ${memalloc[$(($i-1))]} --net $network --ip ${iplist[$(($i-1))]} \
		-d tflogitertime:v1 /bin/bash
	done

	echo "logging start time in re-run loop..."
	./date.sh
	for i in $(seq 1 $num_nodes)
	do
        	service_id=""
        	if [ $(($i-1)) -lt $num_ps ]
 		then
                	service_id="tf-ps-"$(($i-1)) 
        		echo "going to launch resnet model on container $service_id"
        	elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        	then
        		service_id="tf-master-"$(($i-$(($num_ps+1))))
        	else
        		service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
		fi

		echo "going to launch resnet model on container $service_id"
 		if [ "$sync_mode" == "false" ]
		then
			echo "running ASP mode"
	       	docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py \
				--data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --num-gpus=0 \
				--train-steps='${stepsequence[$z]}' --warm-start='true''  &>> $log_dir$run_number$service_id/"stepslogs.txt" &
		elif [ "$sync_mode" == "true" ]
		then
			echo "ALL EXCEPT LAST running in BSP mode as sync is set to true.."
			docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py \
            	--data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --num-gpus=0 \
                --train-steps='${stepsequence[$z]}' --sync --warm-start='true''  &>> $log_dir$run_number$service_id/"stepslogs.txt" &
		fi
	done

	echo "value of z..."
	echo "$((z+1))"
	#blocking call made here to wait for current run to finish
	while true
	do
		#if [[ "cat $logdirectory | tr -cd 'INFO:tensorflow:Loss for final step:' | wc -c" -eq "($z+1)" ]]
		if [[ "$(grep -o "INFO:tensorflow:Loss for final step:" $logdirectory | wc -l)" -eq "$((z+1))" ]]
		then
			echo "finished another re-run...."
			break
		fi
	done

	echo "logging end time in re-run loop..."
	./date.sh
done
