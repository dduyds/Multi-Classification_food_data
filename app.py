import streamlit as st
from models import CTranModel
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import random
import hashlib
from sklearn import metrics

# Đường dẫn dữ liệu
data_root = 'data'
food_root = os.path.join(data_root, 'food')
val_root = os.path.join(food_root, 'val')
val_image_names_dir = os.path.join(val_root, 'image_names.npy')
val_label = os.path.join(val_root, 'labels.npy')

# Hàm tải mô hình
@st.cache_resource
def load_model():
    model = CTranModel(None, None, None, 323, False, False, 2, 4, 0.1, False)
    model_path = 'pretrain_model/food.2layer.bsz_16.adam1e-05.layers2_heads4_noLMT_kn0_bb6_323lbs/best_model.pt'
    test_model = torch.load(model_path)
    model.load_state_dict(test_model['state_dict'])
    return model

# Hàm lấy chỉ số mask unknown
def get_unk_mask_indices(image, testing, num_labels, known_labels):
    if testing:
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels - int(known_labels)))
    else:
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0, int(num_labels * 0.75))
        else:
            num_known = 0
        unk_mask_indices = random.sample(range(num_labels), (num_labels - num_known))
    return unk_mask_indices

# Hàm xử lý ảnh đầu vào
def process_image(image):
    scale_size = 640
    crop_size = 576
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    testTransform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normTransform
    ])
    
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
        pred, int_pred, attns = model(image.cuda().unsqueeze(0), mask_in.cuda().unsqueeze(0))
        rs_pred = torch.sigmoid(pred)
    return rs_pred, label

# Hàm xử lý kết quả
def handle_result(pred, label, food_label):
    meanAP = metrics.average_precision_score(
        label.cpu().numpy(), np.squeeze(pred.detach().cpu().numpy()), average='macro', pos_label=1
    )
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    label_test = label.cuda().clone()
    equal = pred == label_test.view(*pred.shape)
    pred_rs = food_label[np.where(pred.detach().cpu() == 1)[1]]
    target_rs = food_label[np.where(label == 1)]
    accuracy = torch.mean(equal.type(torch.FloatTensor))
    return pred_rs, target_rs, meanAP, accuracy.item()

# CSS cho giao diện màu sắc
st.markdown(
    """
    <style>
    .main {
        background-color: #f5fcfb; /* Màu nền đậm hơn */
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #FF6347;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
    }
    .header {
        color: #4682B4;
        font-size: 2em;
        font-weight: bold;
    }
    .subheader {
        color: #2E8B57;
        font-size: 1.5em;
        font-weight: bold;
    }
    .metric {
        color: #FF4500;
        font-size: 1.2em;
        font-weight: bold;
    }
    .result {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tạo ứng dụng Streamlit
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">Image Classification with C-Tran</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify</div>', unsafe_allow_html=True)

# Tải ảnh từ thư mục
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    labels = np.load(val_label)
    names = np.load(val_image_names_dir)
    
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    food_label = np.load('label_of_food_323.npy')
    
    # Xử lý ảnh và dự đoán lớp
    image_name = uploaded_file.name
    model = load_model()
    processed_image = process_image(image)
    pred, label = predict(processed_image, model, image_name, names, labels)
    pred_rs, target_rs, meanAP, accuracy = handle_result(pred, label, food_label)

    # Hiển thị kết quả
    st.markdown('<div class="subheader">Model Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">Mean Average Precision (mAP): {meanAP:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric">Accuracy (threshold = 0.5): {accuracy:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Classification Results</div>', unsafe_allow_html=True)
    # st.markdown('<div class="metric">Threshold = 0.5</div>', unsafe_allow_html=True)

    # st.markdown(f'<div class="result"><b>Accuracy:</b> {accuracy:.2f} </div>', unsafe_allow_html=True)
    st.markdown('<div class="result"><b>Predicted Labels:</b> ' + ', '.join(pred_rs) + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="result"><b>True Labels:</b> ' + ', '.join(target_rs) + '</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
