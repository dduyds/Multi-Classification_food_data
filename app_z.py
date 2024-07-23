import streamlit as st
from models import CTranModel
import torch
from torchvision import models, transforms
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import random
import hashlib
from sklearn import metrics


data_root = 'data'
food_root = os.path.join(data_root,'food')
val_root = os.path.join(food_root,'val')
val_image_names_dir = os.path.join(val_root,'image_names.npy')
val_label = os.path.join(val_root,'labels.npy')


# Hàm tải mô hình
@st.cache_resource
def load_model():
    model = CTranModel(None, None, None, 323, False, False, 2, 4, 0.1, False)
    model_path = 'pretrain_model/food.2layer.bsz_16.adam1e-05.layers2_heads4_noLMT_kn0_bb6_323lbs/best_model.pt'
    test_model = torch.load(model_path)
    model.load_state_dict(test_model['state_dict'])
    return model

def get_unk_mask_indices(image, testing, num_labels, known_labels):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices

# Hàm xử lý ảnh đầu vào
def process_image(image):
    scale_size = 640
    crop_size = 576
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    testTransform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                    transforms.CenterCrop(crop_size),
                                    transforms.ToTensor(),
                                    normTransform])
    
    return testTransform(image)

# Hàm dự đoán lớp của ảnh
def predict(image, model, image_name, names, labels):
    pos = np.where(names == image_name)[0][0]
    with torch.no_grad():
        label = torch.Tensor(labels[pos])
        unk_mask_indices = get_unk_mask_indices(image, True, 323, 0)
        mask = label.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
        mask_in = mask.clone()
        model.cuda()
        pred, int_pred, attns = model(image.cuda().unsqueeze(0),mask_in.cuda().unsqueeze(0))
        # Make all pred into sigmod function
        rs_pred = torch.sigmoid(pred)
    return rs_pred, label

def handle_result(pred, label, food_label):
    meanAP = metrics.average_precision_score(label.cpu().numpy(), 
                                              np.squeeze(pred.detach().cpu().numpy()), average='macro', pos_label=1)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    label_test = label.cuda().clone()
    equal = pred == label_test.view(*pred.shape) 
    pred_rs = food_label[np.where(pred.detach().cpu() == 1)[1]]
    target_rs = food_label[np.where(label == 1)]
    accuracy = torch.mean(equal.type(torch.FloatTensor))

    return pred_rs, target_rs, meanAP, accuracy.item()

# Tạo ứng dụng Streamlit
st.title('Image Classification with C-Tran')

# Tải ảnh từ thư mục
uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])
if uploaded_file is not None:
    labels = np.load(val_label)
    names = np.load(val_image_names_dir)
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    food_label = np.load('label_of_food_323.npy')

    # Xử lý ảnh và dự đoán lớp
    image_name = uploaded_file.name
    model = load_model()
    processed_image = process_image(image)
    pred, label = predict(processed_image, model, image_name, names, labels)
    pred_rs, target_rs, meanAP, accuracy = handle_result(pred, label, food_label)

    st.title("Model Evaluation Metrics")
    st.markdown(f"Mean Average Precision (mAP): {meanAP:.2f}")

    st.markdown(f"With Threshold = {0.5}:")
    st.write(f"  - Accuracy: {accuracy:.2f}")
    st.write(f"  - Prediction: {pred_rs}")
    st.write(f"  - True label: {target_rs}")

