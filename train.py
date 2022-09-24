from sklearn.random_projection import SparseRandomProjection
from sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import argparse
import shutil
import faiss
import torch
import glob
import cv2
import os

from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter


def gray_blur(img, blur_type, kernel_size):

    img = np.array(img)
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([150, 150, 150])
    upper = np.array([255, 255, 255])

    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur_type == 1:
        blur = cv2.medianBlur(gray, kernel_size)
    else:
        blur = cv2.blur(gray, [kernel_size, kernel_size])

    blur = cv2.bitwise_and(blur, blur, mask=255-mask2)

    result = cv2.bitwise_and(img, img, mask=mask2) + np.transpose(np.array((blur, blur, blur)), (1, 2, 0))
    return Image.fromarray(result)

def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    # make embeddings dir
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, args, root, transform, gt_transform, phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
        self.args = args

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if self.args.blur_type == 1 or self.args.blur_type == 2:
            img = gray_blur(img, self.args.blur_type, self.args.kernel_size)
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)
    

class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []        

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(args, root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=8) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(args, root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=8) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        torch.no_grad()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.index = faiss.read_index(os.path.join(self.embedding_dir_path, "b={}_k={}_index.faiss".format(args.blur_type, args.kernel_size)))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        #faiss
        self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        self.index.add(self.embedding_coreset)
        faiss.write_index(self.index, os.path.join(self.embedding_dir_path, "b={}_k={}_index.faiss".format(args.blur_type, args.kernel_size)))


    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type = batch

        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        score_patches, _ = self.index.search(embedding_test , k=args.n_neighbors)

        score_file = open("./score/b={}_k={}_score.txt".format(args.blur_type, args.kernel_size), 'w')
        for row in score_patches:
            for score in row:
                score_file.write(str(round(score.item(), 4)) + ' ')
            score_file.write('\n')
        score_file.close()
        exit(1)

        if args.score_type == 1:
            score_patches = np.sum(score_patches, axis = 1) / args.n_neighbors
            anomaly_map = score_patches.reshape((28, 28))
            score = max(score_patches) # Image-level score

        elif args.score_type == 2:
            score_patches = ((1 - np.exp(score_patches[:,0]) / np.sum(np.exp(score_patches), axis=1))) * score_patches[:, 0]
            anomaly_map = score_patches.reshape((28, 28))
            score = max(score_patches) # Image-level score

        elif args.score_type == 3:
            anomaly_map = score_patches[:,0].reshape((28,28))
            N_b = score_patches[np.argmax(score_patches[:,0])]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score

        else:
            anomaly_map = score_patches[:,0].reshape((28,28))
            N_b = score_patches[np.argmax(score_patches[:,0])]
            w = (1 - (np.min(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score

        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        result_file = open("./result/s={}_b={}_k={}_v={}_result.txt".format(args.score_type, args.blur_type, args.kernel_size, args.v_num), 'w')
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        result_file.write('pixel_auc: {}\nimg_auc: {}'.format(pixel_auc, img_auc))
        result_file.close()

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test', 'all', 'testall', 'getscore'], default='all')
    parser.add_argument('--dataset_path', default=r'./dataset')
    parser.add_argument('--category', default='capsule')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--project_root_path', default=r'./')
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    parser.add_argument('--score_type', choices=[0, 1, 2, 3], type=int, default=0) # 0: original score, 1: average score, 2: reversed softmax score
    parser.add_argument('--blur_type', choices=[0, 1, 2], type=int, default=0) # 0: no blur, 1: median blur, 2: average blur
    parser.add_argument('--kernel_size', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    if args.phase == 'train':
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0)
        model = STPM(hparams=args)
        trainer.fit(model)
    elif args.phase == 'test':
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0)
        model = STPM(hparams=args)
        trainer.test(model)
    elif args.phase == 'all':
        v_num = 52
        bk_list = [(0, 0),
                (1, 5), (1, 7), (1, 9),
                (1, 11), (1, 13), (1, 15),
                (2, 5), (2, 7), (2, 9),
                (2, 11), (2, 13), (2, 15)]
        arg_dict = vars(args)
        for i in range(3):
            for blur_type, kernel_size in bk_list:
                arg_dict['blur_type'] = blur_type
                arg_dict['kernel_size'] = kernel_size
                arg_dict['v_num'] = v_num
                trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
                model = STPM(hparams=args)
                trainer.fit(model)
                v_num += 1
                for score_type in range(3):
                    arg_dict['score_type'] = score_type
                    arg_dict['v_num'] = v_num
                    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
                    model = STPM(hparams=args)
                    trainer.test(model)
                    v_num += 1
    
    elif args.phase == 'testall':
        v_num = 208
        bk_list = [(0, 0),
                (1, 5), (1, 7), (1, 9),
                (1, 11), (1, 13), (1, 15),
                (2, 5), (2, 7), (2, 9),
                (2, 11), (2, 13), (2, 15)]
        arg_dict = vars(args)
        arg_dict['score_type'] = 3
        for blur_type, kernel_size in bk_list:
            arg_dict['blur_type'] = blur_type
            arg_dict['kernel_size'] = kernel_size
            arg_dict['v_num'] = v_num
            trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
            model = STPM(hparams=args)
            trainer.test(model)
            v_num += 1
    elif args.phase == 'getscore':
        v_num = 208
        bk_list = [(0, 0),
                (1, 5), (1, 7), (1, 9),
                (1, 11), (1, 13), (1, 15),
                (2, 5), (2, 7), (2, 9),
                (2, 11), (2, 13), (2, 15)]
        blur_type, kernel_size = 2, 11
        arg_dict = vars(args)
        arg_dict['score_type'] = 3
        arg_dict['blur_type'] = blur_type
        arg_dict['kernel_size'] = kernel_size
        arg_dict['v_num'] = v_num
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=0) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
        model = STPM(hparams=args)
        trainer.test(model)
        v_num += 1