from __future__ import division
import os
import time
import numpy as np
from six.moves import xrange
import scipy.io as sio
import scipy as sp

import argparse
from sklearn import datasets as ds

from torch.autograd import Variable

import torch

from transOptModel import TransOpt

from utils import *
from trans_opt_objectives import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='MAE_', help='folder name')
parser.add_argument('--steps', type=int, default=10, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--c_dim', type=int, default=1, help='number of color channels in the input image')
parser.add_argument('--z_dim', type=int, default=2, help='Dimension of the latent space')
parser.add_argument('--x_dim', type=int, default=50, help='Dimension of the input space')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--lr_psi', type=float, default=0, help='learning rate for Psi')
parser.add_argument('--b1', type=float, default=0.5, help='adam: momentum term')
parser.add_argument('--b2', type=float, default=0.999, help='adam: momentum term')
parser.add_argument('--gamma', type=float, default=0.00008, help='gd: weight on dictionary element')
parser.add_argument('--zeta', type=float, default=0.000001, help='gd: weight on coefficient regularizer')
parser.add_argument('--to_weight', type=float, default=0.01, help='Weight for transport operator loss')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--M', type=int, default=4, help='Number of dictionary elements')
parser.add_argument('--decay', type=float, default=0.999, help='Decay weight')
parser.add_argument('--data_use', type=str,default = 'gait',help='Specify which dataset to use [concen_circle,rotDigits,gait]')
parser.add_argument('--training_phase', type=str,default = 'finetune',help='Specify which training phase to run [AE_train,transOpt_train,finetune]')
opt = parser.parse_args()
print(opt)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
           
# Define variables from input parameters
batch_size = opt.batch_size
input_h = opt.img_size
input_w = opt.img_size
c_dim = opt.c_dim
z_dim = opt.z_dim
x_dim = opt.x_dim
N = opt.z_dim
N_use = N*N
M = opt.M
zeta = opt.zeta
data_use = opt.data_use



# Parameter for scaling the latent space to accommodate coefficient inference
if opt.training_phase == 'AE_train':
    scale = 1.0
else:
    scale = 10.0

test_size = batch_size

# Specify which classes to train on
class_use = np.array([0,1,2,3,4,5,6,7,8,9])
class_use_str = np.array2string(class_use)    

# Initialize psi learning rate and the maximum psi learning rate
lr_psi = opt.lr_psi
if data_use == 'rotDigits':
    lr_max = 300.0 # Rotated digits, this value
elif data_use == 'gait':
    lr_max = 1.0
else: 
    lr_max = 1.0
decay = opt.decay
np.seterr(all ='print')

# Defin save directories
save_folder = './' +  opt.model + opt.data_use  + '_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_scale' + str(scale) + '_M' + str(M) + '_' + opt.training_phase + '/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if data_use == 'rotDigits':
    sample_dir = './' + opt.model + opt.data_use +  '_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_scale' + str(scale) + '_M' + str(M) + '_' + opt.training_phase + '_samples/'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

# Define the loss functions
mse_loss = torch.nn.MSELoss(reduction = 'mean')
latent_mse_loss = torch.nn.MSELoss(reduction = 'sum')


transNet = TransOpt()

# Load data
nTrain = 4000
nTest = 500
noise_std = 0.01    
if data_use == 'concen_circle':
    from fullyConnectedModel import Encoder
    from fullyConnectedModel import Decoder

    mapMat = np.random.uniform(-1,1,(x_dim,z_dim))
    if opt.training_phase == 'AE_train':
        sio.savemat(save_folder + 'mapMat_circleHighDim_z' + str(z_dim) + '_x' + str(x_dim) +'.mat',{'mapMat':mapMat})
    else:
        mat_folder = './' +  opt.model + opt.data_use  + '_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_scale' + str(1.0) + '_M' + str(M) + '_AE_train/'
        mat_load = sio.loadmat(mat_folder + 'mapMat_circleHighDim_z' + str(z_dim) +  '_x' + str(x_dim) +'.mat')
        mapMat = mat_load['mapMat']
    [noisy_circles,y_circles] = ds.make_circles(n_samples=nTrain, factor=.5,
                                              noise=noise_std)
    data_y = y_circles
    data_X = np.transpose(np.matmul(mapMat,np.transpose(noisy_circles)))
    dist_sample = sp.spatial.distance_matrix(data_X,data_X)
    
    [noisy_circles,y_circles] = ds.make_circles(n_samples=nTest, factor=.5,
                                              noise=noise_std)
    sample_labels = y_circles
    sample_inputs = np.transpose(np.matmul(mapMat,np.transpose(noisy_circles)))
    sample_inputs_torch = torch.from_numpy(sample_inputs)
    sample_inputs_torch = sample_inputs_torch.float()

    encoder = Encoder(x_dim,z_dim)
    decoder = Decoder(x_dim,z_dim)
    data_size = nTrain
elif data_use == 'rotDigits':
    from covNetModel import Encoder
    from covNetModel import Decoder
    class_transform = np.array([0,1,2,3,4,5,6,7,8,9])
    data_X, data_y = load_mnist("val")

    sample_labels = data_y[0:test_size]
    sample_inputs, sample_angles = transform_image(data_X[0:test_size,:,:,:],sample_labels,np.asarray(range(0,10)),input_h,360.0)
    sample_inputs_torch = torch.from_numpy(sample_inputs)
    sample_inputs_torch = sample_inputs_torch.permute(0,3,1,2)
    sample_inputs_torch = sample_inputs_torch.float()
    data_X, data_y = load_mnist("train")
    encoder = Encoder(z_dim,opt.c_dim,opt.img_size)
    decoder = Decoder(z_dim,opt.c_dim,opt.img_size)
    data_size = len(data_X)
elif data_use == 'gait':
    from fullyConnectedLargeModel import Encoder
    from fullyConnectedLargeModel import Decoder
    data_folder = '/home/marissa/Documents/Data/CMU_moCap/subject_035'
    sample_inputs,frameUse_sample = load_gait_data(data_folder,test_size,opt.x_dim,'val') 
    data_y = np.zeros((sample_inputs.shape[0],1)) 
    sample_inputs_torch = torch.from_numpy(sample_inputs)
    sample_inputs_torch = sample_inputs_torch.float()
    encoder = Encoder(x_dim,z_dim)
    decoder = Decoder(x_dim,z_dim)
    data_size = 2
    

# Initialize dictionary
Psi_std = 0.01
normalize_val = 0.02
Psi = Variable(torch.mul(torch.randn(N_use, M, dtype=torch.double),Psi_std), requires_grad=True)
if opt.training_phase == 'AE_train':
    # Initialize network weights
    params_use = list(encoder.parameters()) + list(decoder.parameters()) 
    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
else:
    # Load current set of network weights and transport operator weights
    checkpoint_folder = '/home/marissa/Dropbox/Documents/Tensorflow/autoencoder_transform_pytorch/final_code/MAE_gait_batch32_zdim2_zeta1e-06_gamma8e-05_scale10.0_M4_transOpt_train'
    checkpoint_file = 'network_batch32_zdim2_zeta1e-06_gamma8e-05_step6.pt'
    checkpoint = torch.load(checkpoint_folder + '/' + checkpoint_file)
    encoder.load_state_dict(checkpoint['model_state_dict_encoder'])
    decoder.load_state_dict(checkpoint['model_state_dict_decoder'])
    Psi = checkpoint['Psi']
    params_use = list(encoder.parameters()) + list(decoder.parameters())  + list(transNet.parameters())

Psi_use = Psi.detach().numpy()    

# Specify optimizer
optimizer = torch.optim.Adam(params_use, lr=opt.lr, betas=(opt.b1, opt.b2))

# Initialize arrays for saving progress
start_time = time.time()
counter = 0
loss_total = np.zeros((opt.steps))
loss_auto_0 = np.zeros((opt.steps))
loss_frob = np.zeros((opt.steps))
loss_auto_1 = np.zeros((opt.steps))
loss_trans = np.zeros((opt.steps))
lr_save = np.zeros((opt.steps))

for step in xrange(opt.steps):
    batch_counter = 1
    # Choose random indices for data
    randIdx = np.random.permutation(data_size)[0:opt.batch_size]
    # Load a batch of data
    if opt.data_use == 'concen_circle':
        [batch_input_0,batch_input_1] = get_neighbor_batch(data_X,dist_sample,opt.batch_size)
    elif opt.data_use == 'rotDigits':
        batch_labels = data_y[randIdx]
        batch_input_0,batch_input_1, batch_angles = transform_image_pair(data_X[randIdx],batch_labels,class_transform,input_h,350,5)   
    elif opt.data_use == 'gait':
        batch_input_0,batch_input_1,frame0,frame1 = load_gait_data_pair(data_folder,opt.batch_size,opt.x_dim,15,'train')
    
    # Convert numpy data to torch format
    batch_input_torch_0 = torch.from_numpy(batch_input_0).float()
    if opt.data_use == 'rotDigits':
        batch_input_torch_0 = batch_input_torch_0.permute(0,3,1,2)
        
    if opt.training_phase != 'AE_train':
        batch_input_torch_1 = torch.from_numpy(batch_input_1).float()
        if opt.data_use == 'rotDigits':
            batch_input_torch_1 = batch_input_torch_1.permute(0,3,1,2)

    # Set gradients to zero
    optimizer.zero_grad()
    
    # Encode data in the latent space
    latent_out_0 = encoder(batch_input_torch_0)
    # Compute individual autoencoder loss
    auto_loss_0 = mse_loss(decoder(latent_out_0),batch_input_torch_0)
    
    
    if opt.training_phase != 'AE_train':
        #Encode data into the latent space and scale
        latent_out_0_scale = torch.div(latent_out_0,scale)
        latent_0_np_scale = latent_out_0_scale.detach().numpy()
        latent_out_1 = encoder(batch_input_torch_1)
        latent_out_1_scale = torch.div(latent_out_1,scale)
        latent_1_np_scale = latent_out_1_scale.detach().numpy()

        # Infer coefficients for every pair of points
        c_est = np.zeros((batch_size,M))
        for b in range(0,batch_size):
            x0_single = latent_0_np_scale[b,:]
            x1_single = latent_1_np_scale[b,:]
            c_est_temp = infer_transOpt_coeff(x0_single,x1_single,Psi_use,zeta,0.0,1.0)
            c_est[b,:] = c_est_temp
 
        # Transform the latent vector using the inferred coefficeints
        z1_est = transNet(latent_out_0_scale.double(),torch.from_numpy(c_est),Psi)
        z1_test = z1_est.detach().numpy()
        
        # Compute the autoencode rloss and transport operator loss functions
        auto_loss_1 = mse_loss(decoder(latent_out_1).float(),batch_input_torch_1.float()) 
        t_loss = (0.5*latent_mse_loss(latent_out_1_scale.double(),z1_est)/batch_size + 0.5*opt.gamma*torch.sum(torch.pow(Psi,2)))
        frob_loss = 0.5*opt.gamma*torch.sum(torch.pow(Psi,2))
        # Define the loss functions for the transport operator training and finetuning phases
        if opt.training_phase == 'transOpt_train':
            auto_loss = t_loss
        else:
            auto_loss = auto_loss_0 + auto_loss_1 + opt.to_weight*t_loss.float()
        # Store the original loss values prior to the transport operator gradient step
        loss_val_comp = np.zeros((2))
        Psi_comp = np.zeros((2,N_use,M))    
        loss_val = t_loss.detach().numpy()
        loss_val_comp[0] = loss_val
        Psi_comp[0,:,:] = Psi.detach().numpy()
        
    else:
        auto_loss = auto_loss_0
        
    # Take gradient step on network weights   
    auto_loss.backward()
    optimizer.step()
    
    loss_total[counter] = auto_loss.detach().numpy()
    
    if opt.training_phase != 'AE_train':
        # Take gradient step on transport operators
        Psi_grad = Psi.grad.detach().numpy()
        Psi.data.sub_(lr_psi*Psi.grad.data)
        #Check if the transport operator update improves the transport operator loss
        z1_est_temp = transNet(latent_out_0_scale.double(),torch.from_numpy(c_est),Psi)
        t_loss_new = (0.5*latent_mse_loss(latent_out_1_scale.double(),z1_est_temp)/batch_size  + 0.5*opt.gamma*torch.sum(torch.pow(Psi,2)))
        loss_val_new = t_loss_new.detach().numpy()
        loss_val_comp[1] = loss_val_new
        Psi_comp[1,:,:] = Psi.detach().numpy()
        # If the psi objective after the Psi update is higher than the objective before the update, don't accept the step and decrease lr
                # If the psi objective after the Psi update is lower than the objective before the update, accept the step and increase lr
        if loss_val_comp[1] > loss_val_comp[0]:
            Psi.data.add_(lr_psi*Psi.grad.data)
            lr_psi = lr_psi*decay
            print('Failed Step')
            loss_val_use = loss_val_comp[0]
            if counter > 0:
                loss_trans[counter] = loss_trans[counter-1]
            else: 
                loss_trans[counter] = t_loss.detach().numpy()
        else:
            if lr_psi < lr_max:
                lr_psi = lr_psi/decay
            lr_save[counter] = lr_psi
            loss_val_use = loss_val_comp[1]
            loss_trans[counter] = t_loss.detach().numpy()
        Psi.grad.data.zero_()  
        
        # Save loss terms
        loss_frob[counter] = frob_loss.detach().numpy()
        loss_auto_0[counter] = auto_loss_0.detach().numpy()
        loss_auto_1[counter] = auto_loss_1.detach().numpy()

        print ("[Step %d/%d] time: %4.4f lr: %f [loss 0: %f] [loss 1: %f] [loss total: %f]" % (step, opt.steps,time.time() - start_time,lr_psi,
                                                             loss_val_comp[0]*1000,loss_val_comp[1]*1000,loss_total[counter]*1000))
    else:
        print ("[Step %d/%d] time: %4.4f [loss: %f]" % (step, opt.steps,time.time() - start_time,
                                                             auto_loss.item()))
        
        
        
    # Save intermediate files    
    if np.mod(counter,2) == 0:
        sample_latent = encoder(sample_inputs_torch)
        sample_out = decoder(sample_latent)
        
        if opt.data_use == 'rotDigits':
            # Plot reconstructed outputs
            samples = sample_out[0:16,:,:,:]
            samples = samples.permute(0,2,3,1)
            samples = samples.detach().numpy()
            save_images(samples, [4, 4],
                                '{}train_{:04d}.png'.format(sample_dir, step))
        Psi_new = Psi.detach().numpy()
        # Save network weights
        torch.save({'step': counter,'model_state_dict_encoder': encoder.state_dict(),'model_state_dict_decoder': decoder.state_dict(),
            'model_state_dict_transOpt': transNet.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss_total,'Psi':Psi,
            }, save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '.pt')
    
        # Save loss values
        sio.savemat(save_folder + 'lossVal_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '.mat',{'loss_total':loss_total[:counter],'loss_auto_0':loss_auto_0[:counter],
            'loss_auto_1':loss_auto_1[:counter],'loss_trans':loss_trans[:counter],'Psi_new':Psi_new,'loss_frob':loss_frob[:counter],'lr_max':lr_max});
    if np.mod(counter,3) == 0:
        # Save network weight checkpoints
        torch.save({'step': counter,'model_state_dict_encoder': encoder.state_dict(),'model_state_dict_decoder': decoder.state_dict(),
            'model_state_dict_transOpt': transNet.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss_total,'Psi':Psi,
            }, save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_step' + str(counter) + '.pt')
        # Save loss value and transport operator checkpoints    
        sio.savemat(save_folder + 'lossVal_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_step' + str(counter) + '.mat',{'loss_total':loss_total[:counter],'loss_auto_0':loss_auto_0[:counter],
            'loss_auto_1':loss_auto_1[:counter],'loss_trans':loss_trans[:counter],'Psi_new':Psi_new,'loss_frob':loss_frob[:counter],'params':opt,'sample_latent':sample_latent.detach().numpy(),
            'sample_out':sample_out.detach().numpy(),'sample_inputs':sample_inputs,'lr_max':lr_max,'lr_save':lr_save[:counter]});
        trainParams_save = {'zeta':zeta,'gamma':opt.gamma,'batch_size':opt.batch_size,'Psi_std':Psi_std,'normalize_val':normalize_val,'data_use':data_use,'params':opt,
                                               'lr_psi':lr_psi,'scale':scale,'M':M,'decay':decay,'lr_all':opt.lr,'to_weight':opt.to_weight,'lr_max':lr_max}
        if opt.training_phase != 'AE_train':
            trainParams_save['checkpoint_folder'] = checkpoint_folder
            trainParams_save['checkpoint_file'] = checkpoint_file
        sio.savemat(save_folder + 'trainInfo.mat',trainParams_save)
    counter += 1
    
torch.save({'step': counter,'model_state_dict_encoder': encoder.state_dict(),'model_state_dict_decoder': decoder.state_dict(),
            'model_state_dict_transOpt': transNet.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss_total,'Psi':Psi,
            }, save_folder + 'network_batch' + str(opt.batch_size) + '_zdim' + str(opt.z_dim) + '_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_step' + str(counter) + '.pt')
# Save loss value and transport operator checkpoints    
sio.savemat(save_folder + 'lossVal_zeta' + str(zeta) + '_gamma' + str(opt.gamma) + '_step' + str(counter) + '.mat',{'loss_total':loss_total[:counter],'loss_auto_0':loss_auto_0[:counter],
    'loss_auto_1':loss_auto_1[:counter],'loss_trans':loss_trans[:counter],'Psi_new':Psi_new,'loss_frob':loss_frob[:counter],'params':opt,'sample_latent':sample_latent.detach().numpy(),
    'sample_out':sample_out.detach().numpy(),'sample_inputs':sample_inputs,'lr_max':lr_max,'lr_save':lr_save[:counter]});
