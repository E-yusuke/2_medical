
import matplotlib.pyplot as plt
import cv2


file_list = ["train", "test", "val"]
train_data = []
val_data = []
test_data = []
image_size = (214, 214)

dirname = "C:/Users/nagat/Dev/cv/2_medical/medical_dataset/train/0/"
files = os.listdir(dirname)
for fname in files:  # あとはForで1ファイルずつ実行されていく
    bgr = cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    train_data.append(resized)
train_data = np.array(train_data)

print(train_data.shape)
