
import os


args = [#f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        #f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet ',
        #f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet  --max_points 1000' ,
        f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
]

max_points = 1000
losses = ['LazyQuadrupletLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
                experiment = f'-e Sampling/{loss_func}/P1-N18/'+ str(max_points)

                loss =  f'--loss {loss_func}'
                sampling = f'--max_points {max_points}'
                func_arg = arg + ' ' + loss + ' ' + ' ' + sampling + ' ' +  experiment
                os.system('python3.8 train_knn.py ' + func_arg)