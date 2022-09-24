
score_type = 1 # 0: original score, 1: new score
blur_type = 1 # 0: no blur, 1: median blur, 2: average blur
kernel_size = 15



def gray_blur(img):

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

#########################################################################################################


    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if blur_type:
            img = gray_blur(img)
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type


###########################################################################################3


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
        anomaly_map = score_patches[:,0].reshape((28,28))

        if score_type:
            new_score_patches = np.sum(score_patches, axis = 1) / args.n_neighbors
            anomaly_map = new_score_patches.reshape((28,28))
        
        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
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