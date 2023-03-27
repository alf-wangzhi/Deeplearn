import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

file_path = "data/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
img = Image.open(file_path)
print("origin_img_size:", img.size)

trans = transforms.Compose([transforms.Resize((224, 224))])
trans = transforms.Compose([transforms.Resize((224,224)),

                                     ])

img1 = trans(img)
print("随机裁剪后的大小:", img1.size)
plt.subplot(1, 2, 1), plt.imshow(img)
plt.subplot(1, 2, 2), plt.imshow(img1)
plt.show()
