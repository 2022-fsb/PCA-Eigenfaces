from __future__ import division
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import bunch
import fnmatch
import re
from sklearn.utils import shuffle
import random

#-----------------------------------------------------------------------------------------------------
Test_Dir = ""
test_data = ""
test_response = ""
#-----------------------------------------------------------------------------------------------------
class FaceRecognition(object):
    # load images
    def __init__(self, image_path=Test_Dir, data_test=test_data, response_test=test_response, suffix="*.pgm",
                 variance_pct=0.80, knn=5):
        self.variance_pct = variance_pct # variance percentage
        self.knn = knn
        self.image_story = [] # the original images
        self.data_test = data_test # bring first subject of training
        self.response_test=test_response
        image_names = []
        for root, dirnames, filenames in os.walk(image_path):
            for filename in fnmatch.filter(filenames, suffix):
                image_names.append(os.path.join(root, filename))
        for idx, image_name in enumerate(image_names):
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
            if idx == 0:
                self.imgShape = img.shape # the shape of the image.
                # the normalized image matrix
                self.vector_matrix = np.zeros((self.imgShape[0] * self.imgShape[1], len(image_names)), dtype=np.float64)
            self.image_story.append((image_name, img, self.getClassFromName(image_name)))
            self.vector_matrix[:, idx] = img.flatten()
# -----------------------------------------------------------------------------------------------------
        subjects = set()
        for _, _, subject in self.image_story:
            subjects.add(subject)
        print("loaded all images: %d, subject number is: %d" % (len(self.image_story), len(subjects)))
        self.get_eigen()
        self.getWeight4Training()
# -----------------------------------------------------------------------------------------------------
    def get_eigen(self):
        mean_vector = self.vector_matrix.mean(axis=1)
        for ii in range(self.vector_matrix.shape[1]):
            self.vector_matrix[:, ii] -= mean_vector
        shape = self.vector_matrix.shape
        if (shape[0] < shape[1]):
            _, lamb, u = np.linalg.svd(np.dot(self.vector_matrix, self.vector_matrix.T))
            u = u.T
        else:
            _, lamb, v = np.linalg.svd(np.dot(self.vector_matrix.T, self.vector_matrix))
            v = v.T
            u = np.dot(self.vector_matrix, v)
            norm = np.linalg.norm(u, axis=0)
            u = u / norm
        standard_deviation = lamb ** 2 / float(len(lamb))
        variance_proportion = standard_deviation / np.sum(standard_deviation)
        eigen = bunch.Bunch()
        eigen.lamb = lamb
        eigen.u = u
        eigen.variance_proportion = variance_proportion
        eigen.mean_vector = mean_vector
        self.eigen = eigen
        self.K = self.get_n_components_2_variance(self.variance_pct)
        print("Get the n_components to preserve variance: var=%.2f, K=%d" % (self.variance_pct, self.K))
# -----------------------------------------------------------------------------------------------------
    def getWeight4Training(self):
        self.weightTraining = np.dot(self.eigen.u.T, self.vector_matrix).astype(np.float32)
        return self.weightTraining
# -----------------------------------------------------------------------------------------------------
    def get_eigen_value_distribution(self):
        data = np.cumsum(self.eigen.lamb) / np.sum(self.eigen.lamb)
        return data
# -----------------------------------------------------------------------------------------------------
    def get_n_components_2_variance(self, variance=.95):
        for ii, eigen_value_cumsum in enumerate(self.get_eigen_value_distribution()):
            if eigen_value_cumsum >= variance:
                return ii
# -----------------------------------------------------------------------------------------------------
    def getWeight4NormalizedImg(self, imgNormlized):
        return np.dot(self.eigen.u.T, imgNormlized)
# -----------------------------------------------------------------------------------------------------
    def getWeight4img(self, img):
        return self.getWeight4NormalizedImg(img.flatten - self.eigen.mean_vector)
# -----------------------------------------------------------------------------------------------------
    def reconstruct_eigenFaces(self, img, k=-1):
        if k < 0:
            k = self.K
        ws = self.getWeight4NormalizedImg(img)
        u = self.eigen.u
        imgNew = np.dot(self.eigen.u[:, 0:k], ws[0:k])
        fig, axarr = plt.subplots(1, 2)
        axarr[0].set_title("Original")
        axarr[0].imshow(img.reshape(self.imgShape) + self.get_average_weight_matrix(), cmap=plt.cm.gray)
        axarr[1].set_title("Reconstructed")
        axarr[1].imshow(imgNew.reshape(self.imgShape) + self.get_average_weight_matrix(), cmap=plt.cm.gray)
        return imgNew
# -----------------------------------------------------------------------------------------------------
    def store_training(self, Kpca=-1):
        train_data = self.weightTraining[0:Kpca, :].T.astype(np.float32)
        return train_data
# -----------------------------------------------------------------------------------------------------
    def store_testing(self, Kpca=-1):
        test_data = self.weightTraining[0:Kpca, :].T.astype(np.float32)
        test_data_1 = np.random.randint(1, 3, (len(test_data), 1)).astype(np.float32)
        test_data, test_data_1 = shuffle(test_data, test_data_1)
        return test_data
# -----------------------------------------------------------------------------------------------------
    def check_n_subject(self):
        return self.image_story
# -----------------------------------------------------------------------------------------------------
    def eval(self, knn_k=-1, Kpca=-1):
        response = []
        for name, img, label in self.image_story:
            response.append(label*(random.randint(1,2)*1))
        responses = np.asarray(response, dtype=np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(self.weightTraining[0:Kpca, :].T.astype(np.float32), cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours2, dist = knn.findNearest(self.data_test, knn_k)
        neighbours = neighbours2[:, 1:]
        eval_data = []
        for idx, nb in enumerate(neighbours2):
            neighbours_count = []
            for n in nb:
                neighbours_count.append(nb.tolist().count(n))
            vote = nb[neighbours_count.index(max(neighbours_count))]
            eval_data.append((vote, responses[idx]))
        return eval_data
# -----------------------------------------------------------------------------------------------------
    def get_eval(self, knn_k=-1, Kpca=-1):
        eval_data = self.eval(knn_k, Kpca)
        tp = 0
        fp = 0
        for pair in eval_data:
            if int(pair[0]) == int(pair[1]):
                tp += 1
            else:
                fp += 1
        precision = 1.0 * tp / (tp + fp)
        return precision
# -----------------------------------------------------------------------------------------------------
    def visualize_image_story(self):
        story = self.image_story
        num_row_x = num_row_y = int(np.floor(np.sqrt(len(story) - 1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii, (name, v, _) in enumerate(story):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(v, cmap=plt.cm.gray)
            axarr[div, rem].set_title('{}'.format(self.getClassFromName(name)).capitalize())
            axarr[div, rem].axis('off')
            if ii == len(story) - 1:
                for jj in range(ii, num_row_x * num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')
# -----------------------------------------------------------------------------------------------------
    def visualize_eigen_vector(self, n_eigen=-1, nth=-1):
        if nth is -1:
            self.visualize_eigen_vectors(n_eigen)
        else:
            plt.figure()
            plt.imshow(np.reshape(self.eigen.u[:, nth], self.imgShape), cmap=plt.cm.gray)
# -----------------------------------------------------------------------------------------------------
    def visualize_eigen_vectors(self, number=-1):
        if number < 0:
            number = self.eigen.u.shape[1]
        num_row_x = num_row_y = int(np.floor(np.sqrt(number - 1))) + 1
        fig, axarr = plt.subplots(num_row_x, num_row_y)
        for ii in range(number):
            div, rem = divmod(ii, num_row_y)
            axarr[div, rem].imshow(np.reshape(self.eigen.u[:, ii], self.imgShape), cmap=plt.cm.gray)
            axarr[div, rem].set_title("%.6f" % self.eigen.variance_proportion[ii])
            axarr[div, rem].axis('off')
            if ii == number - 1:
                for jj in range(ii, num_row_x * num_row_y):
                    div, rem = divmod(jj, num_row_y)
                    axarr[div, rem].axis('off')
# -----------------------------------------------------------------------------------------------------
    def get_average_weight_matrix(self):
        return np.reshape(self.eigen.mean_vector, self.imgShape)
# -----------------------------------------------------------------------------------------------------
    def visualize_mean_vector(self):
        fig, axarr = plt.subplots()
        axarr.set_title("Compute and display the mean face.")
        axarr.imshow(self.get_average_weight_matrix(), cmap=plt.cm.gray)
# -----------------------------------------------------------------------------------------------------
    def visualize_pca_components_proportions(self):
        fig, axarr = plt.subplots()
        plt.xlabel('number of components')
        plt.ylabel('Percentage of variance')
        axarr.set_title("Visualize PCA Components")
        axarr.scatter(range(self.eigen.variance_proportion.size), self.eigen.variance_proportion)
# -----------------------------------------------------------------------------------------------------
    def visualize_eigen_value_distribution(self):
        fig, axarr = plt.subplots()
        plt.xlabel('number of components')
        plt.ylabel('Percentage of variance')
        axarr.set_title("Visualize EigenValue Distribution")
        data = np.cumsum(self.eigen.lamb, ) / np.sum(self.eigen.lamb)
        axarr.scatter(range(data.size), data)
# -----------------------------------------------------------------------------------------------------
    def getClassFromName(self, fileName, lastSubdir=True):
        if lastSubdir:
            name = os.path.basename(os.path.dirname(fileName))
        else:
            name = os.path.basename(fileName)
        mat = re.search(".*(\d+).*", name)
        if mat != None:
            return int(mat.group(1))
        else:
            return name.__hash__()
# -----------------------------------------------------------------------------------------------------
