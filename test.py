import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import cv2 
import numpy as np
import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import Dataset
from skimage.metrics import structural_similarity 
from skimage.metrics import peak_signal_noise_ratio
from model import XBDCNN
import glob, time, argparse
np.random.seed(1)



################## add hybrid noise################
def AddHybridNoise(image,sigma=None):
    """input:image: CxHxW
    return: CxHxW
    """
    image = image.data.cpu().numpy()
    assert len(image.shape) == 3, "input image shape"
    c,h,w = image.shape
    if np.max(image)<=1:
        image = image*255.0
    sigma_s = np.random.randint(0,20) #the standard variance of gaussian
    eta = np.random.randint(4,19)/10 # the intensity of poisson

    guassian_noise = np.random.normal(scale=sigma_s,size=(h,w))
    img_poisson = np.random.poisson(image[0]*eta)/eta
    mixed_noise_img = img_poisson + guassian_noise
    mixed_noise_img = np.clip(mixed_noise_img,0,255)/255.
    
    gaussian_level = sigma_s*np.ones_like(image[0])/255.
    poisson_level = eta/np.ones_like(image[0])
    noise_level = np.stack((poisson_level,gaussian_level),axis=0)

    mixed_noise_img = torch.from_numpy(np.expand_dims(mixed_noise_img,0))
    noise_level = torch.from_numpy(np.expand_dims(noise_level,0))

    return mixed_noise_img, noise_level


def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    im1 = im1.data.cpu().numpy().astype(np.float32)
    im2 = im2.data.cpu().numpy().astype(np.float32)
    return im1, im2


def compare_psnr(im_true, im_test, data_range=1.0):
    im_true, im_test = _as_floats(im_true, im_test)
    psnr = 0
    for i in range(im_true.shape[0]):
        psnr += peak_signal_noise_ratio(im_true[i,0,:,:], im_test[i,0,:,:], data_range=data_range)
    
    return (psnr / im_true.shape[0])

def compare_ssim(im_true, im_test, data_range=1.0):
    im_true, im_test = _as_floats(im_true, im_test)
    im_true = np.transpose(im_true,(0,2,3,1))
    im_test = np.transpose(im_test,(0,2,3,1))
    assert im_test.shape == im_true.shape
    ssim = 0
    for n in range(im_true.shape[0]):
        ssim += structural_similarity(im_true[n,:,:,:], im_test[n,:,:,:], multichannel=True)
    aver_ssim = ssim/im_true.shape[0]

    return aver_ssim


def compare_snr(im_true, im_test, data_range=1.0):
    im_true, im_test = _as_floats(im_true, im_test)
    signal = np.sum(np.square(im_true))
    noise = np.sum(np.square(im_test - im_true)) 

    return 10*np.log10(signal / noise)


def test_set(datadir, model_path):   
    """
    test on testing set
    input clean image dataset and synthetic to noisy image,
    output the mean and standard deviation of PSNR, SSIM, SNR
    """

    result = 'result/tested/' 
    if not os.path.exists(result):
        os.makedirs(result)

    # load model
    model_info = torch.load(model_path)
    print('Loading model ...\n')
    net = XBDCNN()
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(model_info["state_dict"])

    # load data
    val_data = Dataset(datadir, False)
    val_data = DataLoader(val_data, batch_size = 32, shuffle=False)

    psnr_test, snr_test, ssim_test = [], [], []
    base_psnr_test = []

    for i, data in enumerate(val_data):
        model.eval()
        target_img = data
        noisy_img = torch.zeros_like(target_img)
        for n in range(target_img.size()[0]):
            noisy_img[n,:,:,:], _ = AddHybridNoise(target_img[n,:,:,:])
        
        target_img = Variable(target_img.cuda())
        noisy_img = Variable(noisy_img.cuda())  # input

        with torch.no_grad():
            _, out = model(noisy_img)
            denoised_img = noisy_img - out
            denoised_img = torch.clamp(denoised_img, 0., 1.)

        psnr = compare_psnr(target_img, denoised_img, data_range=1.)
        ssim = compare_ssim(target_img, denoised_img, data_range=1.)
        snr = compare_snr(target_img, denoised_img, data_range=1.)

        base_psnr = compare_psnr(target_img, noisy_img, 1.)

        base_psnr_test.append(base_psnr)
        psnr_test.append(psnr)
        snr_test.append(snr)
        ssim_test.append(ssim)

        print("The %s batch images (base_psnr %.4f) PSNR/SNR/SSIM %.4f/%.4f/%.4f" % (i, base_psnr, psnr, snr, ssim))
        if i < 10 :
            temp = np.concatenate((noisy_img[0][0].data.cpu().numpy().astype(np.float32),\
            target_img[0][0].data.cpu().numpy().astype(np.float32),\
                denoised_img[0][0].data.cpu().numpy().astype(np.float32)),axis=1)
            temp = (temp*255).astype('uint8')
            cv2.imwrite(result+'denoised_%d.png'%i, temp)


    print("\nPSNR/SNR/SSIM on test data (%.4f) is %.4f/%.4f/%.4f" %(np.mean(base_psnr_test), np.mean(psnr_test),\
        np.mean(snr_test),np.mean(ssim_test)))
    print("PSNR/SNR/SSIM standard deviation on test data is %.4f/%.4f/%.4f"%(np.std(psnr_test), np.std(snr_test),np.std(ssim_test)))
    print("test speed on test data is %.4f image/sec "%((time.time()-start_time)/len(val_data)))
    print("Done")
    

def test_fixed_noise(clean_dir, noisy_dir, model_path):

    """
    test on testing set for fixed noise
    input clean image dataset and add fixed noise to them,
    output the mean and standard deviation of PSNR, SSIM, SNR
    """

    result = 'result/fixed/' 
    if not os.path.exists(result):
        os.makedirs(result)

    # load model
    model_info = torch.load(model_path)
    print('Loading model ...\n')
    net = XBDCNN()
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(model_info["state_dict"])
    model.eval()
    # load data 
    noisy_list = glob.glob(os.path.join(noisy_dir, '*_10_0.8.png'))
    length = len(noisy_list)
    
    print(length)
    psnr_test, snr_test, ssim_test = [], [], []
    base_psnr_test = []
    base_snr_test = []
    base_ssim_test = []

    for img in noisy_list:
        name = img.split('/')[-1].split('_')[0] + ".png"
        clear_img = os.path.join(clean_dir, name)

        savedir = os.path.join(result, img.split('/')[-1].split(".png")[0] + '_denoised.png')

        noisy_img = cv2.imread(img, 0)/255.
        noisy_img = np.expand_dims(noisy_img, 0)
        noisy_img = np.expand_dims(noisy_img, 0)
        noisy_img = torch.Tensor(noisy_img)

        target_img = cv2.imread(clear_img, 0)/255.
        target_img = np.expand_dims(target_img, 0)
        target_img = np.expand_dims(target_img, 0)
        target_img = torch.Tensor(target_img)

        target_img, noisy_img = Variable(target_img.cuda()), Variable(noisy_img.cuda())

        with torch.no_grad():
            _, out = model(noisy_img)
            denoised_img = noisy_img - out
            denoised_img = torch.clamp(denoised_img, 0, 1.)

        psnr = compare_psnr(target_img, denoised_img, data_range=1.)
        ssim = compare_ssim(target_img, denoised_img, data_range=1.)
        snr = compare_snr(target_img, denoised_img, data_range=1.)
        
        base_psnr = compare_psnr(target_img, noisy_img, data_range=1.)
        base_snr = compare_snr(target_img, noisy_img, data_range=1.)
        base_ssim = compare_ssim(target_img, noisy_img, data_range=1.)

        psnr_test.append(psnr)
        snr_test.append(snr)
        ssim_test.append(ssim)

        base_psnr_test.append(base_psnr)
        base_snr_test.append(base_snr)
        base_ssim_test.append(base_ssim)

        # print(img.split('/')[-1], "psnr/snr/ssim %.4f/%.4f/%.4f"%(base_psnr, base_snr, base_ssim))
        # print("psnr/snr/ssim %.4f/%.4f/%.4f"%(psnr, snr, ssim))

        # save images
        denoised_img = denoised_img[0][0].cpu().numpy()
        denoised_img = (denoised_img * 255).astype('uint8')

        cv2.imwrite(savedir, denoised_img)
    print("base psnr snr ssim %.4f/%.4f/%.4f"%(np.mean(base_psnr_test), np.mean(base_snr_test), np.mean(base_ssim)))
    print("denoise psnr snr ssim %.4f/%.4f/%.4f"%(np.mean(psnr_test), np.mean(snr_test), np.mean(ssim_test)))


def test_realnoise(im_dir, model_path):

    """
    test on real noisy image
    input real noise image,
    output the denoised image
    """

    result = '/home/yin/Code/JAS/classification/our_img/'  # save dir 
    if not os.path.exists(result):
        os.makedirs(result)

    # load img
    img_list = []
    if os.path.isdir(im_dir):
        img_list = glob.glob(im_dir + '*.png')
    else:
        img_list.append(im_dir)
    print(len(img_list))

    # load model
    model_info = torch.load(model_path)
    print('Loading model ...\n')
    net = XBDCNN()
    device_ids = [0]

    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(model_info["state_dict"])


    for img in img_list:
        model.eval()

        name = img.split('/')[-1]
        # name = img.split('/')[-1].split('.')[0] + "_denoised.png"
        save_dir = os.path.join(result, name)

        noisy_img = cv2.imread(img, 0)/255.
        noisy_img = np.expand_dims(noisy_img, 0)
        noisy_img = np.expand_dims(noisy_img, 0)
        noisy_img = torch.Tensor(noisy_img)


        noisy_img =  Variable(noisy_img.cuda())

        with torch.no_grad():
            _, out = model(noisy_img)
            denoised_img = noisy_img - out
            denoised_img = torch.clamp(denoised_img, 0, 1.)

        psnr = compare_psnr(noisy_img, denoised_img, data_range=1.)
        ssim = compare_ssim(noisy_img, denoised_img, data_range=1.)
        snr = compare_snr(noisy_img, denoised_img, data_range=1.)
        
        denoised_img = denoised_img[0][0].cpu().numpy()
        denoised_img = (denoised_img * 255).astype('uint8')

        cv2.imwrite(save_dir, denoised_img)

        print(img,end='  ')
        
        print("psnr/snr/ssim %.4f/%.4f/%.4f"%(psnr, snr, ssim))


if __name__ == "__main__":
    start_time = time.time()
    model_path = './checkpoint/checkpoint.pth.tar'
    ### 1. experiments on testing set
    datadir = "./datasets/"
    test_set(datadir, model_path)

    ### 2. experiments on fixed noise 
    clean_dir = "/home/yin/Code/JAS/cleanimg"
    noisy_dir = "/home/yin/Code/JAS/fixednoise"
    test_fixed_noise(clean_dir, noisy_dir, model_path)

    ### 3. experiments on real noisy image
    im_dir = "./classification/noisy_img/"
    test_realnoise(im_dir, model_path)
