#!/bin/sh
#echo $1

VIEW="range bev height"

#echo $VIEW

if [[ $VIEW == *"$1"* ]];then 
    MODALITY=$1
else
    echo "Sorry, I don't understand"
    exit
fi

echo $HOSTNAME
#Adapt training condition base don the machine that is running  
case $HOME in
    "C:\Users\Tiago")
        echo "I'm in $HOME"
        BATCH_SIZE=50
        MINI_BATCH_SIZE=50
        ;;
    "/home/tiago")
        echo "Deep Learning $HOME"
        BATCH_SIZE=20
        MINI_BATCH_SIZE=20
        ;;
    *)
        echo "Sorry, I don't understand"
        exit
        ;;
esac


echo $MODALITY
echo $BATCH_SIZE
echo $MINI_BATCH_SIZE
python3 train_knn.py  --memory RAM --device cuda --resume None --batch_size $BATCH_SIZE --mini_batch_size $MINI_BATCH_SIZE  --modality $MODALITY  --session kitti


