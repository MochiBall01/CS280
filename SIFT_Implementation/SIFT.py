import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
from skimage import feature

# Article that inspired most of code - https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

# configuring matplotlib
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# setting images
eiffel_1 = cv2.imread('eiffel-1.jpg', flags=cv2.IMREAD_GRAYSCALE)
eiffel_2 = cv2.imread('eiffel-2.jpg', flags=cv2.IMREAD_GRAYSCALE)


def display(img, title=None):
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# # testing image displays
# print('displaying image 1')
# display(eiffel_1)
#
# print('displaying image 2')
# display(eiffel_2)


# creates scale space
def scale_space(img, show=None):

    print('creating scale space')

    # creating structure of octaves and blurs
    out = {0: [0, 0, 0, 0, 0],
           1: [0, 0, 0, 0, 0],
           2: [0, 0, 0, 0, 0],
           3: [0, 0, 0, 0, 0]}

    # creating the octaves
    for a in range(4):
        # making deepcopy of image
        new_img = copy.deepcopy(img)

        # computing the shape, taking sequentially smaller fourths
        new_shape = (int(img.shape[1] * ((4-a)/4)), int(img.shape[0] * ((4-a)/4)))
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
                plt.axis('on')
                plt.title(f"octave: {a} blur: {b}")

        plt.show()

    print('scale space completed')

    return out


# Difference of Gaussians
def dog(imgs, show=None):

    print('starting difference of gaussians')

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
                fig = plt.figure(figsize=(10, 7))

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

    print('difference of gaussians completed')

    return out


# finds the local maxima and minima of the image, outputs them as image files
def find_maxima_minima(imgs, show=None, is_list=None):

    print('finding local maxima/minima')

    # makes each to be the size of the most high resolution image
    maxima = np.zeros(imgs[0][0].shape)
    minima = np.zeros(imgs[0][0].shape)

    max_list = []
    min_list = []

    for a in range(len(imgs)):
        for b in range(len(imgs[a])):
            for x in range(len(imgs[a][b])):

                # finds relative position of each pixel (x)
                rel_x = x / len(imgs[a][b])

                for y in range(len(imgs[a][b][x])):

                    # same but for y
                    rel_y = y / len(imgs[a][b][x])

                    # finds position of neighboring pixels in middle octaves
                    if a > 0:
                        if (a+1) < len(imgs[a]):

                            # finding positions for all images
                            # ceil to make sure each input is valid
                            img2 = imgs[a-1][b]
                            x2 = math.floor(rel_x * len(imgs[a-1][b]))
                            y2 = math.floor(rel_y * len(imgs[a-1][b][x2]))

                            img3 = imgs[a+1][b]
                            x3 = math.floor(rel_x * len(imgs[a+1][b]))
                            y3 = math.floor(rel_y * len(imgs[a+1][b][x3]))

                    # finds positions of neighboring pixels in edge octaves
                    if a == 0:
                        img2 = imgs[a+1][b]
                        x2 = math.floor(rel_x * len(imgs[a+1][b]))
                        y2 = math.floor(rel_y * len(imgs[a+1][b][x2]))

                        img3 = None
                        x3 = None
                        y3 = None

                    if a == len(imgs):
                        img2 = imgs[a-1][b]
                        x2 = math.floor(rel_x * len(imgs[a-1][b]))
                        y2 = math.floor(rel_y * len(imgs[a-1][b][x2]))

                        img3 = None
                        x3 = None
                        y3 = None

                    # gathering results
                    results = compare_neighbors(img=imgs[a][b],
                                                x=x,
                                                y=y,
                                                img2=img2,
                                                x2=x2,
                                                y2=y2,
                                                img3=img3,
                                                x3=x3,
                                                y3=y3)

                    if results == 'maxima':
                        maxima[math.floor(rel_x * len(imgs[0][0])),
                               math.floor(rel_y * len(imgs[0][0][0]))] = 1

                        # adds coordinates to list if specified
                        if is_list is True:
                            max_list.append((math.floor(rel_x * len(imgs[0][0])),
                                             math.floor(rel_y * len(imgs[0][0][0]))))

                    if results == 'minima':
                        minima[math.floor(rel_x * len(imgs[0][0])),
                               math.floor(rel_y * len(imgs[0][0][0]))] = 1

                        if is_list is True:
                            min_list.append((math.floor(rel_x * len(imgs[0][0])),
                                             math.floor(rel_y * len(imgs[0][0][0]))))

    # displays max and min maps
    if show is True:
        fig = plt.figure(figsize=(7, 5))

        fig.add_subplot(1, 2, 1)
        plt.imshow(maxima)
        plt.axis('on')
        plt.title("maxima")

        fig.add_subplot(1, 2, 2)
        plt.imshow(minima)
        plt.axis('on')
        plt.title("minima")

        plt.show()

    print('local maxima/minima found')

    if list is True:
        return max_list, min_list

    else:
        return maxima, minima


# function to compare pixel to neighbors, avoids repetitive code
# inputs for three images, third image set to none as not all inputs will be comparing three images
# original code simply finds all images that are greater/smaller
# I have adjusted it to include a threshold here instead of later
def compare_neighbors(img, x, y, img2, x2, y2, img3=None, x3=None, y3=None):
    ismax = False
    ismin = False

    # creates list of values to check point off of
    all_values = []

    # adding values that are guaranteed to work

    if x > 0:
        all_values.append(img[x-1][y])

        if y > 0:
            all_values.append(img[x-1][y-1])
            all_values.append(img[x][y-1])

        if (y+1) < len(img[x]):
            all_values.append(img[x-1][y+1])
            all_values.append(img[x][y+1])

    if (x+1) < len(img):
        all_values.append(img[x+1][y])

        if y > 0:
            all_values.append(img[x+1][y-1])
            all_values.append(img[x][y-1])

        if (y+1) < len(img[x]):
            all_values.append(img[x+1][y+1])
            all_values.append(img[x][y+1])

    # adding values for image 2
    all_values.append(img2[x2][y2])

    if x2 > 0:
        all_values.append(img2[x2-1][y2])

        if y2 > 0:
            all_values.append(img2[x2-1][y2-1])
            all_values.append(img2[x2][y2-1])

        if (y2+1) < len(img2[x2]):
            all_values.append(img2[x2-1][y2+1])
            all_values.append(img2[x2][y2+1])

        if (x2+1) < len(img2):
            all_values.append(img2[x2+1][y2])

            if y2 > 0:
                all_values.append(img2[x2+1][y2-1])
                all_values.append(img2[x2][y2-1])

            if (y2+1) < len(img2[x2]):
                all_values.append(img2[x2+1][y2+1])
                all_values.append(img2[x2][y2+1])

    if img3 is not None:

        all_values.append(img3[x3][y3])

        if y3 > 0:
            all_values.append(img3[x3-1][y3-1])
            all_values.append(img3[x3][y3-1])

        if (y3+1) < len(img3[x3]):
            all_values.append(img3[x3-1][y3+1])
            all_values.append(img3[x3][y3+1])

        if (x3+1) < len(img3):
            all_values.append(img3[x3+1][y3])

            if y3 > 0:
                all_values.append(img3[x3+1][y3-1])
                all_values.append(img3[x3][y3-1])

            if (y3+1) < len(img3[x3]):
                all_values.append(img3[x3+1][y3+1])
                all_values.append(img3[x3][y3+1])

    # print(all_values)
    # print(img[x][y])

    # threshold value
    tmin = 40
    tmax = 45

    # debugging
    # print("min", min(all_values) - img[x][y])
    # print("max", img[x][y] - max(all_values))

    if (min(all_values) - img[x][y]) > tmin:
        ismin = True
        # print('ismin')

    if (img[x][y] - max(all_values)) > tmax:
        ismax = True
        # print('ismax')

    # print(f'ismin: {ismin}')
    # print(f'ismax: {ismax}')

    if ismax is True:
        return 'maxima'
    elif ismin is True:
        return 'minima'
    else:
        return None


# function to remove edge responses - detects edges on one image,
# then checks if those pixels in list are along those edges
# removing them
def remove_edge_responses(img, key_img, show=None):

    print('removing edge responses')

    canny = feature.canny(img, 1, 25, 50)

    final_img = copy.deepcopy(key_img)

    # sets all pixels that are along edges to be 0
    final_img[canny] = 0

    # removes keypoints on location based edges of image
    for a in range(len(final_img)):
        for b in range(len(final_img[a])):
            if final_img[a][b] == 1:
                if a <= 10 or a >= (len(final_img) - 10):
                    final_img[a][b] = 0
                if b <= 10 or b >= (len(final_img[a]) - 10):
                    final_img[a][b] = 0

    if show is True:
        fig = plt.figure(figsize=(15, 5))

        fig.add_subplot(1, 3, 1)
        plt.imshow(key_img)
        plt.axis('on')
        plt.title("original")

        fig.add_subplot(1, 3, 2)
        plt.imshow(canny)
        plt.axis('on')
        plt.title("canny")

        fig.add_subplot(1, 3, 3)
        plt.imshow(final_img)
        plt.axis('on')
        plt.title('updated keypoints')

        plt.show()

    print('edge responses removed')

    return final_img


# calculates keypoint orientation
def calc_orientation(img, keypoint_img):
    print("calculating keypoint orientation")

    # converts keypoint image into list
    key_list = []

    for a in range(len(keypoint_img)):
        for b in range(len(keypoint_img[a])):
            if keypoint_img[a][b] == 1:
                key_list.append((a, b))

    # orientation list, same as above
    orientation_list = []

    # computing for area around each keypoint
    for i in range(len(key_list)):

        # creating bins of angle values
        angles = {
            0: 0,
            20: 0,
            40: 0,
            60: 0,
            80: 0,
            100: 0,
            120: 0,
            140: 0,
            160: 0,
            180: 0,
            200: 0,
            220: 0,
            240: 0,
            260: 0,
            280: 0,
            300: 0,
            320: 0,
            340: 0
        }

        for a in range(-4, 3):
            x = (key_list[i][0]+a)
            y = (key_list[i][1]+a)

            # finding gradients in x and y directions

            gx = int(img[x+1][y]) - int(img[x-1][y])
            gy = int(img[x][y+1]) - int(img[x][y-1])

            # find magnitude and orientation

            magnitude = math.sqrt((gx*gx) + (gy*gy))

            # small fix for keypoint with gx of 0
            if gx == 0:
                gx = 1
                gy += 1

            orientation = math.atan(gy/gx)

            if orientation < 0:
                orientation += (2 * math.pi)

            # converts orientation value to degrees
            orientation = (orientation / math.pi) * 180

            # finds which angle bin to add to, divides by 20, rounds down, multiplies by 20
            angle_bin = (math.floor(orientation / 20)) * 20
            angles[angle_bin] += magnitude

        # finds angle with largest value, assignes keypoint that orientation
        max_angle = max(angles, key=angles.get)

        orientation_list.append(max_angle)

    return key_list, orientation_list


def convert_to_keypoint(loc, ang):
    print("converting keypoints to cv2.KeyPoint")

    out = []

    for a in range(len(loc)):
        keypoint = cv2.KeyPoint(loc[a][1], loc[a][0], 10, ang[a])
        out.append(keypoint)

    return out


# function to organize what other fuctions to call
def SIFT(img, show=None):
    # creates scale space of images
    images = scale_space(img)

    # takes the difference of gaussians of that scale space
    dog_images = dog(images)

    # finds maxima and minima, adds them to images
    key_max_img, key_min_img = find_maxima_minima(dog_images)

    # combines two images
    key_img = np.zeros(key_max_img.shape)

    for a in range(len(key_max_img)):
        for b in range(len(key_max_img[a])):
            if key_max_img[a][b] == 1:
                key_img[a][b] = 1
            elif key_min_img[a][b] == 1:
                key_img[a][b] = 1
            else:
                key_img[a][b] = 0

    # removes edge responses
    keypoint_img = remove_edge_responses(img, key_img)

    # calculates orientation of keypoints
    keypoints, angles = calc_orientation(img, keypoint_img)

    # converts them to cv2.keyPoint
    final_keypoints = convert_to_keypoint(keypoints, angles)

    # displays keypoints
    if show is True:
        fig = plt.figure(figsize=(15, 5))

        fig.add_subplot(1, 2, 1)
        img_1 = cv2.drawKeypoints(img, final_keypoints, img)
        plt.imshow(img_1)
        plt.axis('on')
        plt.title('keypoints')

        fig.add_subplot(1, 2, 2)
        key_img = np.zeros(img.shape)
        for a in range(len(keypoints)):
            key_img[keypoints[a][0]][keypoints[a][1]] = 1

        plt.imshow(key_img)
        plt.axis('on')
        plt.title('keypoints loc only')

        plt.show()

    return final_keypoints


# feature matching, copied from article, incomplete
def feature_match(keypoints1, keypoints2, descriptors1, descriptors2, img1, img2):
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x:x.distance)

    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], img2, flags=2)
    plt.imshow(img3), plt.show()


keypoints1 = SIFT(eiffel_1, show=True)
keypoints2 = SIFT(eiffel_2, show=True)

# currently keypoints are in form that does not work with brute force matcher, but sift implementation is working as intended


# testing on cv2's built in model
# sift = cv2.xfeatures2d.SIFT_create()
#
# keypoints_cv2_1, descriptor_cv2_1 = sift.detectAndCompute(eiffel_1, None)
# keypoints_cv2_2, descriptor_cv2_2 = sift.detectAndCompute(eiffel_2, None)
#
# print("keypoints")
# print(keypoints1, keypoints2)
# print(keypoitns_cv2_1, keypoints_cv2_2)
#
# print('descriptors')
# print(orientation1, orientation2)
# print(descriptor_cv2_1, descriptor_cv2_1)






