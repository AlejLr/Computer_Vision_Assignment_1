import sys
import csv
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import neighbors, preprocessing, svm, linear_model

def read_train_data_csv(filename):
    data_dict = {}
    img_paths = []
    labels = []
    with open(filename, 'r')  as train_csv:
        reader = csv.reader(train_csv)
        next(reader, None) # skip the csv header
        for rows in reader:
            img_paths.append(rows[0])
            labels.append(rows[1])
        return img_paths, labels


def read_test_data_csv(filename):
    data_dict = {}
    img_paths = []
    with open(filename, 'r')  as train_csv:
        reader = csv.reader(train_csv)
        next(reader, None) # skip the csv header
        for rows in reader:
            img_paths.append(rows[0])
        return img_paths

def load_images(img_paths):
    imgs = []
    for path in img_paths:
        img = io.imread(path, as_gray=True)
        img_np = np.array(img[0], dtype=float)
        imgs.append(img_np)
    return imgs

def get_fft(img):
    z = np.fft.fft2(img) ## calculates FFT of image
    q = np.fft.fftshift(z) ## shifts center to u=0, v=0
    mag = np.absolute(q) ## magnitude spectrum
    phase = np.angle(q) ## phase spectrum
    return mag, phase

def box_feature(mag, start_x=0, start_y=0, end_x=400, end_y=640):
    total_mag = 0
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            power = mag[x,y]*mag[x,y]
            total_mag += power
    return total_mag

def get_fft_dataset(ims):
    mags = []
    phases = []
    for im in ims:
        mag, phase = get_fft(im)
        mags.append(mag)
        phases.append(phase)
    return mags, phases

def get_features(mags, phases):
    
    
    features = np.zeros((len(mags), 2))
    
    features = []
    
    n_radial = 8 # number of radial sectors
    n_angular = 8 # number of angular sectors
    mid_band = (0.2, 0.6) # min and max radius for the orientation band
    dc_exclude = 0.02 # exclude low frequency band (center)
    eps = 1e-12 # to avoid division by zero
    
    for mag in mags:
        
        # Power spectrum and normalization
        p = float(np.square(mag))
        p = p / (np.sum(p) + eps)
        
        # Polar coordinates 
        rows, cols = mag.shape
        crow, ccol = rows // 2, cols // 2
        y = np.arange(rows) - crow
        x = np.arange(cols) - ccol
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        R_max = np.max(R)
        r_dc = max(dc_exclude * R_max, 1)
        
        # sum of energy by freq manginuted in each radial sector
        
        
        
         
        
    return features

def plot_fft(mag, phase):
    plt.subplot(1,2,1)
    plt.imshow(np.log10(abs(mag)+1), cmap='gray')   ## scaling magnitude so difference is visible
    plt.title('Magnitude')
    plt.subplot(1,2,2)
    plt.imshow(phase, cmap='gray')
    plt.title('Phase')
    plt.show()

def main():
    test_csv = sys.argv[1]

    train_img_paths, train_labels = read_train_data_csv('train.csv')
    test_img_paths = read_test_data_csv(test_csv)

    train_imgs = load_images(train_img_paths)
    test_imgs = load_images(test_img_paths)

    mags, phases = get_fft_dataset(train_imgs)
    test_mags, test_phases = get_fft_dataset(test_imgs)

    plot_fft(mags[0], phases[0]) # Plots an example of magnitude and phase

    train_features = get_features(mags, phases)
    test_features = get_features(test_mags, test_phases)

    # Classifier
    
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)

    predictions_file = test_csv.replace('.csv', '_predictions.csv')
    with open(predictions_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image_path', 'prediction'])
        for i, pred in enumerate(predictions):
            writer.writerow([test_img_paths[i], pred])

if __name__ == "__main__":
    main()
