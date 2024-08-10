import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2

from REC.REC import BeautyREC
from dataset import InferenceDataset

params = {
    'dim':48,
    'style_dim':48,
    'activ': 'relu',
    'n_downsample':2,
    'n_res':3,
    'pad_type':'reflect'
}

def main(args):
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    makeup_paths = args.makeup_paths
    non_makeup_paths = args.non_makeup_paths


    dataset = InferenceDataset(device, makeup_paths, non_makeup_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = BeautyREC(params).to(device)
    model.load(args.model_path, device)
    model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            nonmakeup_img = data['nonmakeup_img'].to(device)
            makeup_img = data['makeup_img'].to(device)
            makeup_seg = data['makeup_seg'].to(device)
            nonmakeup_seg = data['nonmakeup_seg'].to(device)

            pred = model(nonmakeup_img, makeup_img, makeup_seg, nonmakeup_seg)

            no_image = data['nonmakeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            no_image = (no_image.copy()*255).astype(np.uint8)

            image = data['makeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            image = (image.copy()*255).astype(np.uint8)

            pred = pred[0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            pred = (pred.copy()*255).astype(np.uint8)

            cv2.imwrite(f'{args.save_root}/{i}_nomakeup.png', no_image[:,:,::-1])
            cv2.imwrite(f'{args.save_root}/{i}_makeup.png', image[:,:,::-1])
            cv2.imwrite(f'{args.save_root}/{i}_pred.png', pred[:,:,::-1])
            
            
if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(
        device="cpu",
        makeup_paths=["data/wilddataset/images/makeup/15.jpg"],
        non_makeup_paths=['data/wilddataset/images/non-makeup/15.jpg'],
        model_path="model/checkpoints/BeautyREC.pt",
        save_root="data/pred"
    )

    main(args)
