from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np 


def dump_info(file, text, flag='w'):
    
    now = datetime.now()
    current_time = now.strftime("%d|%H:%M:%S")
    
    f = open('results/' + file,flag)
    
    line = "{}||".format(now)

    if isinstance(text,dict):
        for key, values in text.items():
            line += "{}:{} ".format(key,values)
            
    elif isinstance(text,str):
        line += text
        #f.write(line)
    f.write(line + '\n')
    f.close()
    return(line)





def generate_descriptors(model,val_loader, device):
    model.eval()
    
    dataloader = iter(val_loader)
    tbar = tqdm(range(len(val_loader)), ncols=100)

    prediction_bag = {}
    idx_bag = []
    for batch_idx in tbar:
        input,inx = next(dataloader)
        if device in ['gpu','cuda']:
            input = input.to(device)
            input = input.cuda(non_blocking=True)
        
        if len(input.shape)<4:
            input = input.unsqueeze(0)
            
        if not torch.is_tensor(inx):
            inx = torch.tensor(inx)
        # Generate the Descriptor
        prediction = model(input)
        # Keep descriptors
        for d,i in zip(prediction.detach().cpu().numpy().tolist(),inx.detach().cpu().numpy().tolist()):
            prediction_bag[i] = d
    return(prediction_bag)





def unique2D(input):
    if not isinstance(input,(np.ndarray, np.generic)):
        input = np.array(input)
   
    
    output = []
    for p in input:
        output.extend(np.unique(p))
    #p = np.array([np.unique(p) for p in positive]).ravel()
    output = np.unique(output)
    return(output)