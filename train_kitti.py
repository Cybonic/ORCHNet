
import os


args = [#f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet ',
        f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet ',
        f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
]

<<<<<<< HEAD
<<<<<<< Updated upstream
losses = ['LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
                experiment = f'-e FinalResults\\{modality}\\{loss_func}'
=======
losses = ['MSTMatchLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
                experiment = f'-e TEST_LOSSv4/{modality}/{loss_func}/p10n1'
>>>>>>> Stashed changes
=======
#losses = ['LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss']
losses = ['MSTMatchLoss']
for loss_func in losses:
        for arg in args:
                modality = arg.split(' ')[4]
                experiment = f'-e firstResults\\{modality}\\{loss_func}'
>>>>>>> stash@{1}
                loss =  f'--loss {loss_func}'

                func_arg = arg + ' ' + loss + ' ' + experiment
                os.system('python3.8 train_knn.py ' + func_arg)