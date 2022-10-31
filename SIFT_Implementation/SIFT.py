import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

# configuring matplotlib
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# setting images
img1 = cv2.imread('eiffel-1.jpg', flags=cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('eiffel-2.jpg', flags=cv2.IMREAD_GRAYSCALE)


def display(img, title=None):
    # Show image
    plt.figure(figsize = (5,5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# # testing image displays
# print('displaying image 1')
# display(img1)
#
# print('displaying image 2')
# display(img2)


# creates scale space
def scale_space(img, show=None):

    # creating structure of octaves and blurs
    out = {0: [0, 0, 0, 0, 0],
           1: [0, 0, 0, 0, 0],
           2: [0, 0, 0, 0, 0],
           3: [0, 0, 0, 0, 0]}

    # creating the octaves
    for a in range(4):
        # making deepcopy of image
        new_img = copy.deepcopy(img)

        print(img.shape)
        # computing the shape, taking sequentially smaller fourths
        new_shape = (int(img.shape[1] * ((4-a)/4)), int(img.shape[0] * ((4-a)/4)))
        print(new_shape)
        # resizing image
        img_res = cv2.resize(new_img, new_shape)
        # blurring images in each octave
        out[a][0] = img_res
        out[a][1] = cv2.GaussianBlur(img_res, (15, 15), 10)
        out[a][2] = cv2.GaussianBlur(img_res, (25, 25), 10)
        out[a][3] = cv2.GaussianBlur(img_res, (51, 51), 25)
        out[a][4] = cv2.GaussianBlur(img_res, (75, 75), 50)

    # displays images
    if show is True:
        print('displaying images')

        fig = plt.figure(figsize=(10, 7))
        for a in range(4):
            for b in range(5):
                # adds a subplot at position
                fig.add_subplot(4, 5, (5 * a) + (b + 1))

                # displays image
                plt.imshow(out[a][b])
                plt.axis('off')
                plt.title(f"octave: {a} blur: {b}")

        plt.show()

    return out


# Difference of Gaussians
def dog(imgs, show=None):

    # structure of DoG
    out = {0: [0, 0, 0, 0],
           1: [0, 0, 0, 0],
           2: [0, 0, 0, 0],
           3: [0, 0, 0, 0]}

    for a in range(4):
        # print(a)
        for b in range(len(imgs[a]) - 1):
            # print(b)
            dog1 = imgs[a][b]
            dog2 = imgs[a][b+1]
            new = np.zeros(dog1.shape)
            for x in range(len(dog1)):
                for y in range(len(dog1[x])):
                    new[x][y] = abs(int(dog1[x][y]) - int(dog2[x][y]))

            out[a][b] = new

            # displays all three images
            if show is True:
                fig = plt.figure(figsize=(10,7))

                fig.add_subplot(1, 3, 1)
                plt.imshow(dog1)
                plt.axis('off')
                plt.title("Input 1")

                fig.add_subplot(1, 3, 2)
                plt.imshow(dog2)
                plt.axis('off')
                plt.title("Input 2")

                fig.add_subplot(1, 3, 3)
                plt.imshow(new)
                plt.axis('off')
                plt.title("Output")

                plt.show()

    # displays all images
    if show == 'all':
        fig = plt.figure(figsize=(15, 7))
        for a in range(4):
            for b in range(4):
                # adds a subplot at position
                fig.add_subplot(4, 5, (5 * a) + (b + 1))

                # displays image
                plt.imshow(out[a][b])
                plt.axis('off')
                plt.title(f"octave: {a} difference of: {b}, {b+1}")

        plt.show()

    return out


# finds the local maxima and minima of the image, outputs them as image files
# Not yet working, will implement for final exploration
def find_maxima_minima(imgs, show=None):
    # makes each to be the size of the most high resolution image
    maxima = np.zeros(imgs[0][0])
    minima = np.zeros(imgs[0][0])

    for a in range(len(imgs)):
        for b in range(len(imgs[a])):
            for x in range(len(imgs[a][b])):
                for y in range(len(imgs[a][b][x])):
                    # compares pixel to octave above and below if a is greater than zero
                    if a > 0:
                        # scales up image position so it is still accurate



# creates scale space of images
images = scale_space(img1)

# takes the difference of gaussians of that scale space
dog_images = dog(images)






