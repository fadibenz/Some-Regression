# Question 4 Stochastic Grad Descent
import scipy
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
training=scipy.io.loadmat('data.mat')
trainfeat=training['X']
trainlabel=training['y']
num_feat=len(trainfeat[0])
num_train=len(trainfeat)
# Appending extra features and labels to training and test sets
extra_feat=np.ones((num_train,1))
train_appended=np.append(trainfeat,extra_feat,axis=1)
train_appended=np.append(train_appended,trainlabel,axis=1)
num_feat_tot=num_feat+1
# Validation set
np.random.seed(0)
np.random.shuffle(train_appended)
vali_size=1000
validation_set=train_appended[:vali_size]
train_set=train_appended[vali_size:num_train]
# Normalizing all the features
means=[np.mean(train_set[:,i]) for i in np.arange(num_feat)]
stds=[np.std(train_set[:,i]) for i in np.arange(num_feat)]
for i in np.arange(num_feat):
    train_set[:,i]=train_set[:,i] - means[i]
    train_set[:, i] = train_set[:, i] / stds[i]
for i in np.arange(num_feat):
    validation_set[:, i] = validation_set[:, i] - means[i]
validation_set[:, i] = validation_set[:, i] / stds[i]


# Log reg function
def logis(w, X):
    num_samp = X.shape[0]
    s = np.zeros((num_samp,))
    for i in np.arange(num_samp):
        s[i] = np.true_divide(1, 1 + np.exp(-np.dot(X[i], w)))
    return s
# Question 3 SGD with constant learning rate
train_size = num_train - vali_size
# Stochastic grad descent on training set
# For constant learning rate
w = np.zeros((num_feat_tot,))
gradJ = np.zeros((num_feat_tot,))
# For proportional learning rate
w_s = np.zeros((num_feat_tot,))
gradJ_s = np.zeros((num_feat_tot,))
# For constant learning rate, use (for example)
learn_step = 1e-6
# for proportional learning rate, use initial (for example)
learn_step_ini = 1e-4
# regularization paramter
regu_const = 0.1
num_iter = 7000
cost = np.zeros((num_iter + 1,))  # for constant learning rate
cost_s = np.zeros((num_iter + 1,))  # for proportional learning rate
s = logis(w, train_set[:, :num_feat_tot])  # for constant learing rate
ss = logis(w, train_set[:, :num_feat_tot])  # for proportional learning rate
# cost for constant learning rate
cost[0] = -np.dot(train_set[:, num_feat_tot], np.log(s)) - np.dot(
    (1 - train_set[:, num_feat_tot]),
    np.log(1 - s)) + (regu_const / 2) * np.sum(np.square(w))
# cost for proportional learning rate
cost_s[0] = -np.dot(train_set[:, num_feat_tot], np.log(ss)) - np.dot(
    (1 - train_set[:, num_feat_tot]),
    np.log(1 - ss)) + (regu_const / 2) * np.sum(np.square(w_s))
sample_index = -1
for ite in np.arange(num_iter):
    print(ite)
learn_step_s = np.true_divide(learn_step_ini, ite + 1)
sample_index = sample_index + 1
if sample_index == train_size:
    np.random.shuffle(train_set)
sample_index = 0
diff = train_set[:, num_feat_tot] - s  # for constant learning rate
diff_s = train_set[:, num_feat_tot] - ss  # for proportional learning rate
# for constant learning rate


gradJ= regu_const*w - train_size * diff[sample_index]*np.transpose(
train_set[sample_index,:num_feat_tot])
# for proportional learning rate
gradJ_s = regu_const*w_s - train_size * diff_s[sample_index]*np.transpose(
train_set[sample_index,:num_feat_tot])
w=w-learn_step*gradJ # for constant learning rate
w_s=w_s-learn_step_s*gradJ_s # for proportional learning rate
s=logis(w,train_set[:,:num_feat_tot]) # for constant learning rate
ss=logis(w_s,train_set[:,:num_feat_tot]) # for proportional learning rate
# cost for constant learning rate
cost[ite+1]=-np.dot(train_set[:,num_feat_tot],np.log(s))-np.dot(
(1-train_set[:,num_feat_tot]),
np.log(1-s))+(regu_const/2)*np.sum(np.square(w))
# cost for proportional learning rate
cost_s[ite+1]=-np.dot(train_set[:,num_feat_tot],np.log(ss))-np.dot(
(1-train_set[:,num_feat_tot]),
np.log(1-ss))+(regu_const/2)*np.sum(np.square(w_s))
# Plotting cost vs. iterations
plt.plot(np.arange(num_iter+1),cost)
plt.xlabel('Number of iterations in training')
plt.ylabel('’Cost at the end of training')
plt.title('’Training loss vs. Number of iterations for Stochastic Gradient Descent’')
plt.savefig('’Wine_SGD.png')
plt.clf()
plt.close()
# Plotting cost vs. iterations
plt.plot(np.arange(num_iter+1), cost, "-r", label="no decay")
plt.plot(np.arange(num_iter+1),cost_s, "-b", label="decay")
plt.xlabel('’Number of iterations in training’')
plt.ylabel('’Cost at the end of training’')
plt.legend(loc="upper left")
plt.title('’SGD Training loss vs. iterations with decaying & const learning rate’')
plt.savefig('’Wine_SGD_combined.png')
plt.show()


# Checking on validation set
s_test=logis(w,validation_set[:,:num_feat_tot])
ss_test=logis(w_s,validation_set[:,:num_feat_tot])
diffe=np.rint(s_test)-validation_set[:,num_feat_tot]
diffe_s=np.rint(ss_test)-validation_set[:,num_feat_tot]
accu=(np.true_divide(diffe.size-np.count_nonzero(diffe),vali_size))*100
accu_s=(np.true_divide(diffe_s.size-np.count_nonzero(diffe_s),vali_size))*100
print("SGD Validation Accuracy (constant learning rate) is %.2f%%" % (accu))
print("SGD Validation Accuracy (decaying learning rate) is %.2f%%" % (accu_s))