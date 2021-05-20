#!/usr/bin/env python3
import sys
import os
import method.io as io
import method.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

"""
Treat this as a multi-label classification problem, using a cost-sensitive
neural network for imbalanced classification.

Intro to MLC
https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
Cost-sensitive
https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/
Imbalanced
https://machinelearningmastery.com/what-is-imbalanced-classification/
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
(XBG: https://stackoverflow.com/questions/40916939/xgboost-for-multilabel-classification)
https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""

parser = argparse.ArgumentParser('Training AE-multi-label classifier')
parser.add_argument('-s', '--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-a', '--analyse', action='store_true',
                    help='Output analysis results')
parser.add_argument('-d', '--data', type=str, choices=['tp53', 'mlh1', 'abeta'],
                    default='tp53', help='Data for testing the method')
parser.add_argument('-m', '--method', type=str,
                    choices=['pca', 'ae', 'aerf'],
                    default='ae', help='Method for dimension reduction')
parser.add_argument('-c', '--centroid', action='store_true',
                    help='Use centroid only to perform the classification')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Making and showing some plots')
parser.add_argument('-x', '--cached', action='store_true',
                    help='Use cached AE')
args = parser.parse_args()

# NOTE: Perhaps when decided to use this approach, do this as a model selection
#       problem with k-fold validation.
n_pcs = 10

# Make save directory
savedir = 'out/mlc'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = str(args.seed) + '-nlat' + str(n_pcs)

if args.data == 'tp53':
    # Load data
    x, l, m = io.load_training_rama('data/TP53')

    # Split data
    s_seed = args.seed
    #s_seed = 1
    x_train, x_test, l_train, l_test = train_test_split(
        x, list(zip(l, m)), test_size=0.2, random_state=s_seed, shuffle=True
    )
    l_train, m_train = list(zip(*l_train))
    l_test, m_test = list(zip(*l_test))
    l_train, m_train = np.asarray(list(l_train)), list(m_train)
    l_test, m_test = np.asarray(list(l_test)), list(m_test)

    xtrs = x_train.shape  # [-1, 334, 217*2]
    xtes = x_test.shape  # [-1, 334, 217*2]

    # Reshape data
    x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])
    x_test = x_test.reshape(xtes[0] * xtes[1], xtes[2])

    del(x, l, m)
elif args.data == 'mlh1':
    # Load data
    x, l, m = io.load_training_rama('data/MLH1', postfix='_30_40ns')

    # Split data
    s_seed = args.seed
    #s_seed = 1
    x_train, x_test, l_train, l_test = train_test_split(
        x, list(zip(l, m)), test_size=0.2, random_state=s_seed, shuffle=True
    )
    l_train, m_train = list(zip(*l_train))
    l_test, m_test = list(zip(*l_test))
    l_train, m_train = np.asarray(list(l_train)), list(m_train)
    l_test, m_test = np.asarray(list(l_test)), list(m_test)

    xtrs = x_train.shape  # [-1, 334, 217*2]
    xtes = x_test.shape  # [-1, 334, 217*2]

    # Reshape data
    x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])
    x_test = x_test.reshape(xtes[0] * xtes[1], xtes[2])

    del(x, l, m)
elif args.data == 'abeta':
    import pandas
    abeta = pandas.read_csv('data/abeta2.csv')
    l = np.zeros(abeta.shape[0])
    l[abeta.phenotype1 == 'Pathogenic'] = 1
    x = np.asarray(np.asarray(abeta)[:, 3:], dtype=float)
    test = (abeta.grouping == 'E22D') | (abeta.grouping == 'A21G')
    train = ~test
    groups = list(set(abeta.grouping))
    x_train = x[train]
    y_train = l[train]
    x_test = x[test]
    y_test = l[test]

# Transform data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Set seed
np.random.seed(args.seed)

# Dimension reduction
if args.method == 'pca':
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA()
    pca = pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
elif args.method == 'ae':
    # Autoencoder
    import method.autoencoder as autoencoder
    autoencoder.tf.random.set_seed(args.seed)
    encoder_units = [1000, 1000]
    l1l2 = 1e-5
    dropout = 0.3
    lag = 1
    encoder = autoencoder.Encoder(n_components=n_pcs, units=encoder_units)
    if args.cached:
        # Load trained AE
        encoder.load('%s/ae-%s' % (savedir, saveas))
    else:
        # Train AE
        encoder.fit(x_train, lag=lag, shape=xtrs)
        # Save trained AE
        encoder.save('%s/ae-%s' % (savedir, saveas))
    x_train = encoder.transform(x_train, whiten=False)
    x_test = encoder.transform(x_test, whiten=False)
elif args.method == 'aerf':
    # Autoencoder for e.g. 100 features; RF to pick e.g. 10 features
    import method.autoencoder as autoencoder
    n_compression = 100  # something smaller than the full MD features
    autoencoder.tf.random.set_seed(args.seed)
    encoder = autoencoder.Encoder(n_components=n_compression)
    encoder.fit(x_train)
    x_train = encoder.transform(x_train)
    x_test = encoder.transform(x_test)
    # Save trained NN
    encoder.save('%s/aerf-%s' % (savedir, saveas))
    # NOTE, to load:
    # >>> encoder = autoencoder.Encoder(n_components=n_compression)
    # >>> encoder.load('%s/ae-%s' % (savedir, saveas))

    # Randoming AE compressed features with RF
    ms_train = []
    for m in range(len(m_train)):
        ms_train += [m] * xtrs[1]  # times number of MD frames

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import accuracy_score
    rf = RandomForestClassifier(n_estimators=50)
    rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(
        x_train, ms_train, test_size=0.25, random_state=args.seed, shuffle=True
    )
    rf.fit(rf_x_train, rf_y_train)
    #sorted_idx = rf.feature_importances_.argsort()
    perm_importance = permutation_importance(rf, rf_x_test, rf_y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    rf_y_pred = rf.predict(rf_x_test)
    #print(confusion_matrix(rf_y_test, rf_y_pred))
    #print(classification_report(rf_y_test, rf_y_pred))
    print('RF acc. score:', accuracy_score(rf_y_test, rf_y_pred))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 17))
        plt.barh(
            np.asarray(['dim'+str(i + 1)
                        for i in range(n_compression)])[sorted_idx],
            #rf.feature_importances_[sorted_idx]
            perm_importance.importances_mean[sorted_idx]
        )
        plt.xlabel("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig(savedir + '/aerf-importance', dpi=200)
        plt.close()

    x_train = x_train[:, sorted_idx[:n_pcs]]
    x_test = x_test[:, sorted_idx[:n_pcs]]

# Transform data
scaler2 = StandardScaler()
scaler2.fit(x_train)
x_train = scaler2.transform(x_train)
x_test = scaler2.transform(x_test)

# Make y as label * #MD frames
if args.data in ['tp53', 'mlh1']:
    if not args.centroid:
        y_train = []
        for l in l_train:
            y_train += [l[0, 0]] * xtrs[1]  # times #MD frames per variant
        y_train = np.asarray(y_train)
        y_test = []
        for l in l_test:
            y_test += [l[0, 0]] * xtes[1]  # times #MD frames per variant
        y_test = np.asarray(y_test)
    else:  # Compute centroid
        if args.method == 'pca':
            x_train = x_train.reshape(xtrs)
            x_test = x_test.reshape(xtes)
        elif args.method in ['ae', 'aerf']:
            x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))
            x_test = x_test.reshape(xtes[:-1] + (n_pcs,))
        x_train = np.mean(x_train, axis=1)
        x_test = np.mean(x_test, axis=1)
        y_train = l_train[:, 0, 0]
        y_test = l_test[:, 0, 0]

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

#"""
# Try SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

'''
over = SMOTE(sampling_strategy=0.5, k_neighbors=10)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

x_train_2, y_train_2 = pipeline.fit_resample(x_train, y_train)

y_train_2 = np.asarray([[0, 1] if y else [1, 0] for y in y_train_2])
'''
over = SMOTE()
x_train_2, y_train_2 = over.fit_resample(x_train, y_train)

y_train_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_train_2])
#'''

if args.plot:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    b = np.array(y_train[:, 1], dtype=bool)
    x_train_b = x_train[~b].reshape(-1, n_pcs)
    x_train_p = x_train[b].reshape(-1, n_pcs)
    b2 = np.array(y_train_2[:, 1], dtype=bool)
    x_train_b2 = x_train_2[~b2].reshape(-1, n_pcs)
    x_train_p2 = x_train_2[b2].reshape(-1, n_pcs)
    d = np.array(y_test[:, 1], dtype=bool)
    x_test_b = x_test[~d].reshape(-1, n_pcs)
    x_test_p = x_test[d].reshape(-1, n_pcs)
    print(len(x_train_b), len(x_train_p))
    print(len(x_train_b2), len(x_train_p2))
    if not args.centroid:
        skipp = 10
    else:
        skipp = 1

    _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i == j:
                axes[i, j].hist(x_train_p[::, j], color='r', alpha=0.4)
                axes[i, j].hist(x_train_b[::, j], color='b', alpha=0.4)
                axes[i, j].hist(x_test_p[::, j], color='C4', alpha=0.4)
                axes[i, j].hist(x_test_b[::, j], color='C2', alpha=0.4)
            elif i > j:
                axes[i, j].scatter(x_train_p[::skipp, j], x_train_p[::skipp, i],
                                   color='r', alpha=0.4)
                axes[i, j].scatter(x_train_b[::skipp, j], x_train_b[::skipp, i],
                                   color='b', alpha=0.4)
                axes[i, j].scatter(x_test_p[::skipp, j], x_test_p[::skipp, i],
                                   color='C4', alpha=0.4)
                axes[i, j].scatter(x_test_b[::skipp, j], x_test_b[::skipp, i],
                                   color='C2', alpha=0.4)
            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')

            # Set tick labels
            if i < n_pcs - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])
        if i > 0:
            axes[i, 0].set_ylabel('dim %s' % (i + 1))
        else:
            axes[i, 0].set_ylabel('Counts')
        axes[-1, i].set_xlabel('dim %s' % (i + 1))
    plt.suptitle('Train: Blue (SMOTE Benign), Red (Pathogenic)', fontsize=18)
    plt.tight_layout()
    plt.savefig(savedir + '/' + args.method + '-reduction-smote-fig1', dpi=200)
    plt.close()

    _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i == j:
                axes[i, j].hist(x_train_p2[::, j], color='C1', alpha=0.4)
                axes[i, j].hist(x_train_b2[::, j], color='C0', alpha=0.4)
                axes[i, j].hist(x_test_p[::, j], color='C4', alpha=0.4)
                axes[i, j].hist(x_test_b[::, j], color='C2', alpha=0.4)
            elif i > j:
                axes[i, j].scatter(x_train_p2[::skipp, j], x_train_p2[::skipp, i],
                                   color='C1', alpha=0.4)
                axes[i, j].scatter(x_train_b2[::skipp, j], x_train_b2[::skipp, i],
                                   color='C0', alpha=0.4)
                axes[i, j].scatter(x_test_p[::skipp, j], x_test_p[::skipp, i],
                                   color='C4', alpha=0.4)
                axes[i, j].scatter(x_test_b[::skipp, j], x_test_b[::skipp, i],
                                   color='C2', alpha=0.4)
            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')

            # Set tick labels
            if i < n_pcs - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])
        if i > 0:
            axes[i, 0].set_ylabel('dim %s' % (i + 1))
        else:
            axes[i, 0].set_ylabel('Counts')
        axes[-1, i].set_xlabel('dim %s' % (i + 1))
    plt.suptitle('Train: Blue (SMOTE Benign), Red (Pathogenic)', fontsize=18)
    plt.tight_layout()
    plt.savefig(savedir + '/' + args.method + '-reduction-smote-fig2', dpi=200)
    plt.close()
#"""

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

# MLC for B and P
epochs = 100
batch_size = 512
if args.centroid:
    weights = {0:1, 1:1}
else:
    weights = {0:10, 1:1} #{0:100, 1:1}
model = nn.build_dense_mlc_model(input_neurons=n_pcs,
                                 #input_neurons=1024,
                                 #input_neurons=128,
                                 input_dim=n_pcs,
                                 architecture=[128, 128, 128],
                                 #architecture=[1024, 1024],
                                 act_func="leaky_relu",
                                 l1l2=None,  # NOTE: l1l2 matters!
                                 dropout=0.2,  # NOTE: dropout rate matters!
                                 learning_rate=0.001)
model.fit(
    x_train_2[:, :n_pcs],
    y_train_2,
    class_weight=weights,
    epochs=epochs,
    batch_size=batch_size,
)

# Save trained MLC
model.save('%s/mlc-%s' % (savedir, saveas), save_format='tf')
# NOTE, to load:
# >>> model = nn.tf.keras.models.load_model('%s/mlc-%s' % (savedir, saveas)

# Predict
if not args.centroid:
    if args.method == 'pca':
        x_train = x_train.reshape(xtrs)
        x_test = x_test.reshape(xtes)
    elif args.method in ['ae', 'aerf']:
        x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))
        x_test = x_test.reshape(xtes[:-1] + (n_pcs,))

print('Truth   Guess   P   p(B)   p(P)')
pred_test = []
pred_prob_test = []
for x, l in zip(x_test, l_test[:, 0, 0, 1]):
    if not args.centroid:
        pred = model.predict(x[:, :n_pcs])
    else:
        pred = model.predict(x[:n_pcs].reshape(1,-1))
    prob_b = np.mean(pred[:, 0])
    prob_p = np.mean(pred[:, 1])
    #prob_b = np.percentile(pred[:, 0], 75)
    #prob_p = np.percentile(pred[:, 1], 50)
    #prob = np.max(nn.tf.nn.softmax([prob_b, prob_p]).numpy())
    prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob, '  ', prob_b, '  ', prob_p)

    pred_test.append(guess)
    pred_prob_test.append(prob)

    if False:
        import matplotlib.pyplot as plt
        plt.hist(pred[:, 0], bins=np.linspace(0, 1, 25), label='b', density=True)
        plt.hist(pred[:, 1], bins=np.linspace(0, 1, 25), label='p', density=True)
        plt.legend()
        plt.xlabel('Probability')
        plt.ylabel('PDF')
        plt.show()
        plt.close()

print('\nTruth   Guess   P   p(B)   p(P)')
pred_train = []
pred_prob_train = []
for x, l in zip(x_train, l_train[:, 0, 0, 1]):
    if not args.centroid:
        pred = model.predict(x[:, :n_pcs])
    else:
        pred = model.predict(x[:n_pcs].reshape(1,-1))
    prob_b = np.mean(pred[:, 0])
    prob_p = np.mean(pred[:, 1])
    #prob_b = np.percentile(pred[:, 0], 75)
    #prob_p = np.percentile(pred[:, 1], 50)
    #prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
    prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob, '  ', prob_b, '  ', prob_p)

    pred_train.append(guess)
    pred_prob_train.append(prob)


#sys.exit()
#
# VUS
#
if args.data in ['tp53', 'mlh1'] and True:
    if args.data == 'tp53':
        x_vus, m_vus = io.load_vus_rama('data/TP53')
    elif args.data == 'mlh1':
        x_vus, m_vus = io.load_vus_rama('data/MLH1', postfix='_30_40ns')

    xvus = x_vus.shape  # [-1, 334, 217*2]

    x_vus = x_vus.reshape(xvus[0] * xvus[1], xvus[2])

    x_vus = scaler.transform(x_vus)
    if args.method == 'pca':
        x_vus = pca.transform(x_vus)
    elif args.method == 'ae':
        x_vus = encoder.transform(x_vus)
    elif args.method == 'aerf':
        x_vus = encoder.transform(x_vus)
        x_vus = x_vus[:, sorted_idx[:n_pcs]]
    x_vus = scaler2.transform(x_vus)

    x_vus = x_vus.reshape(xvus[:-1] + (n_pcs,))

    pred_vus = []
    pred_prob_vus = []
    for x in x_vus:
        if not args.centroid:
            pred = model.predict(x[:, :n_pcs])
        else:
            pred = model.predict(x[:n_pcs].reshape(1,-1))
        prob_b = np.mean(pred[:, 0])
        prob_p = np.mean(pred[:, 1])
        #prob_b = np.percentile(pred[:, 0], 75)
        #prob_p = np.percentile(pred[:, 1], 50)
        #prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
        prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
        # Unknown or Deleterious
        guess = 'U' if prob_b > prob_p else 'D'

        pred_vus.append(guess)
        pred_prob_vus.append(prob)

    x_vus_c = np.mean(x_vus, axis=1)

if args.plot and (n_pcs == 1):
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 1000)
    plt.plot(x, np.exp(kde_p.score_samples(x.reshape(-1, 1))))
    plt.plot(x, np.exp(kde_b.score_samples(x.reshape(-1, 1))))
    plt.xlabel('PC1')
    plt.ylabel('Probability Density')
    plt.show()

# Compute centroid
if args.data in ['tp53', 'mlh1']:
    if args.centroid:
        x_train_c = x_train
        x_test_c = x_test
    else:
        if args.method == 'pca':
            x_train = x_train.reshape(xtrs)
            x_test = x_test.reshape(xtes)
        elif args.method in ['ae', 'aerf']:
            x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))
            x_test = x_test.reshape(xtes[:-1] + (n_pcs,))

        if args.plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib import cm
            b = np.array(l_train[:, 0, 0, 1], dtype=bool)
            cb = [cm.Blues(x) for x in np.linspace(0.4, 1, len(x_train[b]))]
            cp = [cm.Reds(x) for x in np.linspace(0.4, 1, len(x_train[~b]))]
            d = np.array(l_test[:, 0, 0, 1], dtype=bool)
            cd = [cm.Purples(x) for x in np.linspace(0.7, 1, len(x_test[d]))]
            cu = [cm.Oranges(x) for x in np.linspace(0.7, 1, len(x_test[~d]))]
            _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
            x_train_b = x_train[~b].reshape(-1, n_pcs)
            x_train_p = x_train[b].reshape(-1, n_pcs)
            for i in range(n_pcs):
                for j in range(n_pcs):
                    #"""
                    if i == j:
                        for xi, cbi in zip(x_train[b], cb):
                            axes[i, j].hist(xi[::6, j], color=cbi, alpha=0.5)
                        for xi, cbi in zip(x_test[d], cd):
                            axes[i, j].hist(xi[::6, j], color=cbi, alpha=0.5)
                        for xi, cpi in zip(x_train[~b], cp):
                            axes[i, j].hist(xi[::6, j], color=cpi, alpha=0.5)
                        for xi, cpi in zip(x_test[~d], cu):
                            axes[i, j].hist(xi[::6, j], color=cpi, alpha=0.5)
                    elif i > j:
                        for xi, cbi in zip(x_train[b], cb):
                            axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cbi,
                                               alpha=0.5)
                        for xi, cbi in zip(x_test[d], cd):
                            axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cbi,
                                               alpha=0.5)
                        for xi, cpi in zip(x_train[~b], cp):
                            axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cpi,
                                               alpha=0.5)
                        for xi, cpi in zip(x_test[~d], cu):
                            axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cpi,
                                               alpha=0.5)
                        """
                    if i == j:
                        sns.kdeplot(x=x_train_b[::10, i], ax=axes[i, j])
                        sns.kdeplot(x=x_train_p[::10, i], ax=axes[i, j])
                    elif i > j:
                        sns.kdeplot(x=x_train_b[::15, j], y=x_train_b[::15, i],
                                    ax=axes[i, j])
                        sns.kdeplot(x=x_train_p[::15, j], y=x_train_p[::15, i],
                                    ax=axes[i, j])
                    #"""
                    elif i < j:
                        # Top-right: no plot
                        axes[i, j].axis('off')

                    # Set tick labels
                    if i < n_pcs - 1:
                        # Only show x tick labels for the last row
                        axes[i, j].set_xticklabels([])
                    if j > 0:
                        # Only show y tick labels for the first column
                        axes[i, j].set_yticklabels([])
                if i > 0:
                    axes[i, 0].set_ylabel('dim %s' % (i + 1))
                else:
                    axes[i, 0].set_ylabel('Counts')
                axes[-1, i].set_xlabel('dim %s' % (i + 1))
            plt.suptitle('Train: Red (Benign), Blue (Pathogenic) |'
                         + ' Test: Orange (B), Purple (P)', fontsize=18)
            plt.tight_layout()
            plt.savefig(savedir + '/' + args.method + '-reduction', dpi=200)
            plt.close()
            #plt.show()
        x_train_c = np.mean(x_train, axis=1)
        x_test_c = np.mean(x_test, axis=1)
        if args.plot:
            _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
            for i in range(n_pcs):
                for j in range(n_pcs):
                    if i == j:
                        axes[i, j].hist(x_train_c[b][:, j], color=cb[-1], alpha=0.5)
                        axes[i, j].hist(x_test_c[d][:, j], color=cd[-1], alpha=0.5)
                        axes[i, j].hist(x_train_c[~b][:, j], color=cp[-1], alpha=0.5)
                        axes[i, j].hist(x_test_c[~d][:, j], color=cu[-1], alpha=0.5)
                    elif i > j:
                        for xi, cbi in zip(x_train_c[b], cb):
                            axes[i, j].scatter(xi[j], xi[i], color=cbi, alpha=0.5)
                        for xi, cbi in zip(x_test_c[d], cd):
                            axes[i, j].scatter(xi[j], xi[i], color=cbi, alpha=0.5)
                        for xi, cpi in zip(x_train_c[~b], cp):
                            axes[i, j].scatter(xi[j], xi[i], color=cpi, alpha=0.5)
                        for xi, cpi in zip(x_test_c[~d], cu):
                            axes[i, j].scatter(xi[j], xi[i], color=cpi, alpha=0.5)
                    elif i < j:
                        # Top-right: no plot
                        axes[i, j].axis('off')

                    # Set tick labels
                    if i < n_pcs - 1:
                        # Only show x tick labels for the last row
                        axes[i, j].set_xticklabels([])
                    if j > 0:
                        # Only show y tick labels for the first column
                        axes[i, j].set_yticklabels([])
                if i > 0:
                    axes[i, 0].set_ylabel('dim %s' % (i + 1))
                else:
                    axes[i, 0].set_ylabel('Counts')
                axes[-1, i].set_xlabel('dim %s' % (i + 1))
            plt.suptitle('Train: Red (Benign), Blue (Pathogenic) |'
                         + ' Test: Orange (B), Purple (P)', fontsize=18)
            plt.tight_layout()
            plt.savefig(savedir + '/' + args.method + '-reduction-centroids',
                        dpi=200)
            plt.close()
elif args.data == 'abeta':
    raise NotImplementedError

if args.analyse or True:
    # Output coordinates of the centroids for each mutant
    import pandas as pd
    d_train = {
        'mutants': m_train,
        'B/P': ['P' if i else 'B' for i in l_train[:, 0, 0, 1]],
        'U/D': pred_train,
        'probability': pred_prob_train,
    }
    for i in range(x_train_c.shape[1]):
        d_train['dim' + str(i + 1)] = x_train_c[:, i]
    cols = ['mutants', 'B/P'] + ['dim' + str(i + 1)
                                 for i in range(x_train_c.shape[1])] \
           + ['U/D', 'probability']
    df = pd.DataFrame(d_train, columns=cols)
    df.to_csv('%s/%s-train-%s.csv' % (savedir, args.method, saveas),
              index=False, header=True)

    d_test = {
        'mutants': m_test,
        'B/P': ['P' if i else 'B' for i in l_test[:, 0, 0, 1]],
        'U/D': pred_test,
        'probability': pred_prob_test,
    }
    for i in range(x_test_c.shape[1]):
        d_test['dim' + str(i + 1)] = x_test_c[:, i]
    cols = ['mutants', 'B/P'] + ['dim' + str(i + 1)
                                 for i in range(x_test_c.shape[1])] \
           + ['U/D', 'probability']
    df = pd.DataFrame(d_test, columns=cols)
    df.to_csv('%s/%s-test-%s.csv' % (savedir, args.method, saveas),
              index=False, header=True)

    d_vus = {
        'mutants': m_vus,
        'B/P': ['-' for i in m_vus],
        'U/D': pred_vus,
        'probability': pred_prob_vus,
    }
    for i in range(x_vus_c.shape[1]):
        d_vus['dim' + str(i + 1)] = x_vus_c[:, i]
    cols = ['mutants', 'B/P'] + ['dim' + str(i + 1)
                                 for i in range(x_vus_c.shape[1])] \
           + ['U/D', 'probability']
    df = pd.DataFrame(d_vus, columns=cols)
    df.to_csv('%s/%s-vus-%s.csv' % (savedir, args.method, saveas),
              index=False, header=True)
