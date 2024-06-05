import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
import random



class causal_score(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(causal_score, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch


    def forward(self, x, masks_input, class_p):
        x=x[0]
        self.masks =  masks_input.reshape(-1, 1, *self.input_size)
        self.N = self.masks.shape[0]
        N = self.N 
        H=self.input_size[0]
        W=self.input_size[1]
        masks=self.masks
        # Generate the inverse of masks, i.e., 1-M_i
        masks_inverse=torch.from_numpy(np.repeat((1-self.masks.cpu().numpy())[:, :, np.newaxis,:], 3, axis=2)).cuda()
        
        # Generate the random Gaussian noise
        random_whole=torch.randn([N]+list((3,H,W))).cuda()* 0.1
        
        # Define tensors holding the mask images with noise and original images with noise 
        mask_image_with_noise=torch.empty((N,3,H,W)).cuda()
        original_image_with_noise=torch.empty((N,3,H,W)).cuda()

        for i in range(N):
            # noise to add: Gaussian noise*(1-M_i)
            noise_to_add=random_whole[i]*masks_inverse[i]
            temp_mask=masks[i]
            #thres=np.percentile(temp_mask.cpu().numpy(),50)
            #temp_mask_thres=temp_mask>thres
            mask_image_with_noise[i]=x*temp_mask+noise_to_add
            original_image_with_noise[i]=x+noise_to_add
        
        # Get prediction score for mask images with noise and original images with noise 
        stack_whole=torch.cat((mask_image_with_noise, original_image_with_noise), 0).cuda()
        p_whole = []
        for i in range(0, 2*N, self.gpu_batch):
            p_whole.append(self.model(stack_whole[i:min(i + self.gpu_batch, 2*N)]))
        p_whole = torch.cat(p_whole)
        p_mask_image_with_noise=p_whole[:N]
        p_original_image_with_noise=p_whole[N:]

        # Compute the final causal impact score
        CL = p_mask_image_with_noise.size(1)
        masks_divide=masks/torch.sum(masks,axis=0)
        p_final=p_mask_image_with_noise.data.transpose(0, 1)-p_original_image_with_noise.data.transpose(0, 1)+class_p
        sal = torch.matmul(p_final,masks_divide.view(N,H*W))
        sal = sal.view((CL, H, W))
        sal = sal / N 
        sal = sal.cpu()
        return sal

      