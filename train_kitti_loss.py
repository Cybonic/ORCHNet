
import os


models = ['VLAD_pointnet',
          'GeM_pointnet',
          'VLAD_resnet50',
          'SPoC_resnet50',
          'GeM_resnet50'
        ]
args = [#f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        #f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet',
        f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50',
]
#]

#losses = ['v1','v2','v3','v4','v5','v6','v7']
#losses =  ['v8']
loss_func = 'MetricLazyQuadrupletLoss'
alpha = 0.4
#for version in losses:
for arg in args:
        #for model in models:
                for alpha in range(400,1000,50):
        #        arg = f'--memory RAM  --modality pcl  --session kitti --model {model}'
                        alpha = alpha/1000
                        modality = arg.split(' ')[4]
                        experiment = f'-e NewLosses'
                        #loss =  f'--loss {loss_func}_{version}'
                        loss =  f'--loss {loss_func}'
                        alpha_arg = f'--loss_alpha {float(alpha)}'
                        func_arg = arg + ' ' + loss + ' ' + experiment + ' ' + alpha_arg  + ' ' + '--mini_batch_size' + ' ' + str(10) + ' ' + '--batch_size' + ' ' + str(10)
                        #print(func_arg)
                        os.system('python3.8 train_knn.py ' + func_arg)