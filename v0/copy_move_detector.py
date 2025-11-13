import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from collections import Counter
import os


def readImage(image_name):
    return cv2.imread(str(image_name))


def showImage(image):
    image = imutils.resize(image, width=600)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def featureExtraction(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def featureMatching(keypoints, descriptors):
    norm = cv2.NORM_L2
    k = 10
    bf_matcher = cv2.BFMatcher(norm)
    matches = bf_matcher.knnMatch(descriptors, descriptors, k)

    ratio = 0.5
    good_matches_1, good_matches_2 = [], []

    for match in matches:
        k = 1
        while match[k].distance < ratio * match[k + 1].distance:
            k += 1
        for i in range(1, k):
            if pdist(np.array([keypoints[match[i].queryIdx].pt,
                                keypoints[match[i].trainIdx].pt]), "euclidean") > 10:
                good_matches_1.append(keypoints[match[i].queryIdx])
                good_matches_2.append(keypoints[match[i].trainIdx])

    points_1 = [m.pt for m in good_matches_1]
    points_2 = [m.pt for m in good_matches_2]

    if len(points_1) > 0:
        pts = np.hstack((points_1, points_2))
        unique_pts = np.unique(pts, axis=0)
        return np.float32(unique_pts[:, 0:2]), np.float32(unique_pts[:, 2:4])

    return None, None


def filterOutliers(cluster, points):
    cluster_count = Counter(cluster)
    remove = [c for c in cluster_count if cluster_count[c] <= 1]  # ✅ allow small clusters

    mask = np.array([c not in remove for c in cluster])
    cluster = cluster[mask]
    points = points[mask]

    return cluster, points


def hierarchicalClustering(points_1, points_2, metric, threshold):
    points = np.vstack((points_1, points_2))
    dist_matrix = pdist(points, metric='euclidean')
    Z = hierarchy.linkage(dist_matrix, metric)

    cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4)
    cluster, points = filterOutliers(cluster, points)

    n = int(points.shape[0] / 2)
    return cluster, points[:n], points[n:]


def plotImage(img, p1, p2, C, save_path):
     # Create figure explicitly and don't show it
    fig = plt.figure(figsize=(12, 9))
    # plt.tight_layout(pad=2.0) 
    plt.imshow(img)
    plt.axis('off')

    colors = C[:len(p1)]
    plt.scatter(p1[:, 0], p1[:, 1], c=colors, s=10)

    for (x1, y1), (x2, y2) in zip(p1, p2):
        plt.plot([x1, x2], [y1, y2], 'yellow','c', linestyle=":", linewidth=0.6)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=250)
    plt.close(fig)  # Close the figure instead of clf()


def detect_copy_move(image_path, output_dir="final_results"):
    os.makedirs(output_dir, exist_ok=True)

    image = readImage(image_path)
    kp, desc = featureExtraction(image)
    p1, p2 = featureMatching(kp, desc)

    if p1 is None:
        return False

    clusters, p1c, p2c = hierarchicalClustering(p1, p2, 'ward', 1.5)  # ✅ softer threshold

    if len(clusters) == 0:
        return False

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + "_copymove.png")
    plotImage(img_rgb, p1c, p2c, clusters, out_path)

    return True, out_path
