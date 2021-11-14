import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def display_img(img, mask, mask_cmap=None):

    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(img)
    plt.imshow(mask[:,:,0], alpha=0.6, cmap=mask_cmap)

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(mask[:,:,0], cmap=mask_cmap)

    plt.show()

def hist_val(img, mask):
  
    plt.figure(figsize=(10,5))
    img_expnd = np.reshape(img, img.shape[0]*img.shape[1]*img.shape[2])
    plt.subplot(1,2,1)
    plt.title("IMG val hist")
    plt.hist(img_expnd)
    
    mask_expnd = np.reshape(mask, mask.shape[0]*mask.shape[1])
    print("* # of mask categories:", set(mask_expnd))
    plt.subplot(1,2,2)
    plt.title("Mask val hist")  
    plt.hist(mask_expnd)

    plt.show()


def plot_leraning_curve(H):
    N = len(H.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N),  H.history["loss"], label="train_loss", marker='o',)
    plt.plot(np.arange(0, N),  H.history["val_loss"], label="val_loss", marker='o')
    plt.plot(np.arange(0, N),  H.history["accuracy"], label="train_acc", marker='o')
    plt.plot(np.arange(0, N),  H.history["val_accuracy"], label="val_acc", marker='o')
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("lerning_curve")
    plt.show()
