import os
import cv2
import numpy as np
from matplotlib import pyplot as plt



def getImagePaths(directory):
    imagePaths = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            imagePaths.append(os.path.join(directory, filename))

    return imagePaths


def getImages(imagePaths, downscale=False):
    images = []
    for directory in imagePaths:
        image = cv2.imread(directory)
        if downscale:
            h, w, c = image.shape
            if w >= 2500:
                image = cv2.pyrDown(image)
            if w >= 5000:
                image = cv2.pyrDown(image)
            if w >= 10000:
                image = cv2.pyrDown(image)
        images.append((directory, image))

    return images


class Sample:
    def __init__(self, path, image, keypoints, descriptors):
        self.path = path
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors


def main():
    print("Loading images...")
    queryImagePaths = getImagePaths("query")
    databaseImagePaths = getImagePaths("database")
    queryImages = getImages(queryImagePaths)
    databaseImages = getImages(databaseImagePaths, True)
    querySamples = []
    databaseSamples = []

    sift = cv2.xfeatures2d.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()

    print("Generating keypoints...")
    for image in queryImages:
        keypoints, descriptors = sift.detectAndCompute(image[1], None)
        querySamples.append(Sample(image[0], image[1], keypoints, descriptors))

    for image in databaseImages:
        keypoints, descriptors = sift.detectAndCompute(image[1], None)
        databaseSamples.append(Sample(image[0], image[1], keypoints, descriptors))

    print("Detecting best match...")
    results = []
    for i, querySample in enumerate(querySamples):
        perSampleResults = []
        for j, databaseSample in enumerate(databaseSamples):
            matches = flann.knnMatch(querySample.descriptors, databaseSample.descriptors, k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > 0:
                src_pts = np.float32([querySample.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([databaseSample.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)[1]
                perSampleResults.append((i, j, good, mask))

        perSampleResults.sort(key=lambda x: len(x[3]), reverse=True)
        results.append(perSampleResults)

    if not os.path.exists("results"):
        os.mkdir("results")

    for i, result in enumerate(results):
        figure = plt.figure(querySamples[result[0][0]].path)

        img1 = querySamples[result[0][0]].image
        img2 = databaseSamples[result[0][1]].image
        kp1 = querySamples[result[0][0]].keypoints
        kp2 = databaseSamples[result[0][1]].keypoints
        good = result[0][2]
        mask = result[0][3].ravel().tolist()

        result1 = cv2.drawMatches(img1, [], img2, [], [], None, (0, 255, 255), (0, 255, 255), None, 0)
        result2 = cv2.drawMatches(img1, kp1, img2, kp2, [], None, (0, 255, 255), (0, 255, 255), None, 0)
        result3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, (0, 255, 255), (0, 255, 255), None, 2)
        result4 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, (0, 255, 255), (0, 255, 255), mask, 2)

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Query Image + Best Match")

        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("SIFT Keypoints")

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(result3, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Matches")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result4, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Matches + RANSAC")

        cv2.imwrite("results/img{}_1.jpg".format(i+1), result1)
        cv2.imwrite("results/img{}_2.jpg".format(i+1), result2)
        cv2.imwrite("results/img{}_3.jpg".format(i+1), result3)
        cv2.imwrite("results/img{}_4.jpg".format(i+1), result4)

        plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()