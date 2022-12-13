
import os


args = [#f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet ',
        #f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet ',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
]

<<<<<<< HEAD
#losses = ['LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss']
=======
>>>>>>> tash_issue
losses = ['MSTMatchLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
<<<<<<< HEAD
                experiment = f'-e FinalResults\\{modality}\\{loss_func}'
=======
                experiment = f'-e TEST_LOSSv4/{modality}/{loss_func}/p10n1'
>>>>>>> tash_issue
                loss =  f'--loss {loss_func}'

                func_arg = arg + ' ' + loss + ' ' + experiment
                os.system('python3.8 train_knn.py ' + func_arg)