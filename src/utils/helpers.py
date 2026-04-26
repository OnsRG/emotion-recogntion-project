import json 
import matplotlib.pyplot as plt

def read_json(json_path):
    file = open(json_path)
    data = json.load(file)
    file.close()
    return data

def  save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def show_image_plt(image, bw=False):
    if bw:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')  
    plt.show()