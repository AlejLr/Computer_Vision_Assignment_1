import sys
import csv
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from sklearn import neighbors, preprocessing, metrics, svm, linear_model, ensemble, naive_bayes

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
    mid_band = (0.2, 0.8) # min and max radius for the orientation band
    dc_exclude = 0.02 # exclude low frequency band (center)
    eps = 1e-12 # to avoid division by zero
    
    for i, mag in enumerate(mags):
        
        # Power spectrum and normalization
        p = np.square(mag).astype(np.float64)
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
        ring_edges = np.linspace(r_dc, R_max, n_radial + 1)
        radial_features = np.array([p[(R >= ring_edges[j]) & (R < ring_edges[j + 1])].sum() 
                                    for j in range(n_radial)], dtype=np.float64)
        
        # Orientation band energy
        r_min = mid_band[0] * R_max
        r_max = mid_band[1] * R_max
        band_mask = (R >= r_min) & (R < r_max)
          
        theta_edges = np.linspace(-np.pi, np.pi, n_angular + 1)
        angular_features = np.array([p[band_mask & (Theta >= theta_edges[k]) & (Theta < theta_edges[k + 1])].sum() 
                                     for k in range(n_angular)], dtype=np.float64)

        # Smooth angular features
        angular_features = np.convolve(angular_features, np.ones(3)/3, mode='same')

        ''' 
        Variables for the ratios for features with n_angular = 8
        For n_angular = 8, the edges are [-pi, -3pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3pi/4, pi]
        So the orientation features are:
        Horizontal = around 0 and 180 degrees
        Vertical = around +- 90 degrees
        Diagonal = around +- 45 degrees
        
        This would not work for other n_angular values
        '''
        
        E_h = angular_features[0] + angular_features[4] + angular_features[7]
        E_v = angular_features[2] + angular_features[6]
        E_d = angular_features[3] + angular_features[5]
        
        # Orientation ratios
        ratio_vert_hor = (E_v + eps) / (E_h + eps)
        ratio_diag = (E_d + eps) / (E_h + E_v + eps)
        
        
        # Low and High energy ratios

        E_low = p[R < (0.30 * R_max)].sum()
        E_high = p[R > (0.60 * R_max)].sum()
        
        ratio_low_high = (E_low + eps) / (E_high + eps)
        ratio_high_total = (E_high + eps) / (p.sum() + eps)

        # Phase based features
        phi = phases[i]
        phi_norm = (phi - np.min(phi)) / (np.max(phi) - np.min(phi) + eps)
        
        #phi_var = np.var(phi_norm)
        
        # Phase entropy
        hist, _ = np.histogram(phi_norm, bins=30, range=(0, 1), density=True)
        phi_entropy = -np.sum(hist * np.log(hist + eps))

        feature_vector = np.concatenate([
            radial_features,
            angular_features,
            np.array([ratio_vert_hor, ratio_diag, 
                      ratio_low_high, ratio_high_total, 
                      phi_entropy])
        ])
        

        features.append(feature_vector)

    return np.vstack(features)

def evaluate_classifiers(X_train, train_labels, X_test):
    
    # Evaluates multiple classifiers and prints their accuracy
    
    test_labels = ['S'] * 5 + ['T'] * 5 + ['V'] * 5
    
    KNN = {
        "KNN (1, uniform)": neighbors.KNeighborsClassifier(n_neighbors=1, weights='uniform', p=2),
        "KNN (2, distance)": neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance', p=2),
        "KNN (3, distance)": neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', p=1),
        "KNN (5, uniform)": neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2),
        "KNN (7, uniform)": neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', p=2)
    }
    
    SVM = {
        "SVM (linear)": svm.SVC(kernel='linear', C=1),
        "SVM (rbf)": svm.SVC(kernel='rbf', C=1, gamma='scale'),
        "SVM (poly)": svm.SVC(kernel='poly', C=1, degree=3, gamma='scale')
    }
    LR = {
        "LR-1": linear_model.LogisticRegression(max_iter=1000, C=1.0, penalty='l2'),
        "LR-2": linear_model.LogisticRegression(max_iter=1000, C=0.5, penalty='l2'),
        "LR-3": linear_model.LogisticRegression(max_iter=1000, C=2.0, penalty='l2'),
        "LR-4": linear_model.LogisticRegression(max_iter=1000, C=1.0, penalty='l1', solver='liblinear')
    }
    RF = {
        "RF-1": ensemble.RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, max_features='sqrt', min_samples_leaf=1),
        "RF-2": ensemble.RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, max_features='sqrt', min_samples_leaf=2),
        "RF-3": ensemble.RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, max_features=0.5, min_samples_leaf=1),
        "RF-4": ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, max_features='log2', min_samples_leaf=1)
    }
    NB = {
        "NB-1": naive_bayes.GaussianNB(var_smoothing=1e-9),
        "NB-2": naive_bayes.GaussianNB(var_smoothing=1e-8),
        "NB-3": naive_bayes.GaussianNB(var_smoothing=1e-10)
    }

    classifiers = {"K-NearestNeighbors": KNN, "Support Vector Machine": SVM, "Logistic Regression": LR, "Random Forest": RF, "Naive Bayes": NB}

    for model_name, model in classifiers.items():

        print("\n" + str(model_name) + " Evaluation \n")

        for name, clf in model.items():
            clf.fit(X_train, train_labels)
            preds = clf.predict(X_test)
            
            acc = metrics.accuracy_score(test_labels, preds)
            print(f"{name:20s} Accuracy: {acc*100:.2f}%")
        
        print("\n-------------------------\n")
    
    

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

    # plot_fft(mags[0], phases[0]) # Plots an example of magnitude and phase

    train_features = get_features(mags, phases)
    test_features = get_features(test_mags, test_phases)

    # Classifier
    
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(test_features)

    # Use this to evaluate multiple classifiers
    # The result was that 
    #evaluate_classifiers(X_train, train_labels, X_test)

    
    clf = svm.SVC(kernel='poly', C=1, degree=3, gamma='scale')
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)
    
    
    # Prediction output file

    predictions_file = test_csv.replace('.csv', '_predictions.csv')
    with open(predictions_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['image_path', 'prediction'])
        for i, pred in enumerate(predictions):
            writer.writerow([test_img_paths[i], pred])


if __name__ == "__main__":
    main()
