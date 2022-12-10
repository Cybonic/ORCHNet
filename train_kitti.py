
import os


args = [f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet ',
        #f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet ',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
]

losses = ['MetricLazyQuadrupletLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
                experiment = f'-e testLoss_rpmprop\\{modality}\\{loss_func}\\alpha:0.5'
                loss =  f'--loss {loss_func}'

                func_arg = arg + ' ' + loss + ' ' + experiment
                os.system('python3.8 train_knn.py ' + func_arg)