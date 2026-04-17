import json 
import matplotlib.pyplot as plt

def read_json(json_path):
    file = open(json_path)
    data = json.load(file)
    file.close()
    return data

def show_image_plt(image, bw=False):
    if bw:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')  
    plt.show()