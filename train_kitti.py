
import os


args = [#f'--memory RAM  --modality pcl  --session kitti --model VLAD_pointnet',
        #f'--memory RAM  --modality pcl  --session kitti --model SPoC_pointnet ',
        #f'--memory RAM  --modality pcl  --session kitti --model GeM_pointnet ',
        #f'--memory RAM  --modality pcl  --session kitti --model MAC_pointnet ',
        f'--memory RAM  --modality pcl  --session kitti --model MuHA_pointnet',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model MuHA_resnet50',
]

#losses = ['LazyTriplet_plus','LazyTripletLoss','LazyQuadrupletLoss']
#losses = ['LazyTripletLoss','LazyQuadrupletLoss']
losses = ['LazyTripletLoss']

#density = ['500','1000','5000','10000','20000','30000']
density = ['500','1000','5000','10000','20000','30000']
for loss_func in losses:
        
        for max_point in density:
                for arg in args:
                # modality = arg.split(' ')[4]
                        experiment = f'-e RESULTS-PCLdensity-F1024D2048-ROI-10-'+'P'+str(max_point)
                        #modality = arg.split(' ')[4]
                        #experiment = f'-e sampling\\repeat\\P1-N18-NO-TNET_F64v3-P10000\\'+str(i)
                        loss =  f'--loss {loss_func}'

                        sampling = f'--max_points {max_point}'
                        func_arg = arg + ' ' + loss + ' ' + ' ' + sampling + ' ' +  experiment
                        os.system('python3 train_knn.py ' + func_arg)