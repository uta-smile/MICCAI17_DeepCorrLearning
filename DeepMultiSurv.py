#!/usr/bin/python

import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lifelines.utils import concordance_index
from lasagne.regularization import regularize_layer_params, l1, l2
import pandas as pd

class DeepMultiSurv:
    def __init__(self, learning_rate, channel, width, height, image_pretrain_name, clinical_pretrain_name, clinical_dim=5,
    lr_decay = 0.01, momentum = 0.9,
    L2_reg = 0.0, L1_reg = 0.0,
    standardize = False
    ):
        self.X =  T.ftensor4('x') # patients covariates
        self.E =  T.ivector('e') # the observations vector
        self.Clinical = T.matrix('c')
        # Default Standardization Values: mean = 0, std = 1
        # self.offset = theano.shared(np.zeros(shape = n_in, dtype=np.float32))
        # self.scale = theano.shared(np.ones(shape = n_in, dtype=np.float32))

    ################################ construct network #############################
        self.l_in = lasagne.layers.InputLayer(
            shape=(None, channel, width, height), input_var=self.X
        )
        self.network = lasagne.layers.Conv2DLayer(
            self.l_in,
            num_filters=32,
            filter_size=7,
            stride=3,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))
        self.network = lasagne.layers.Conv2DLayer(
            self.network,
            num_filters=32,
            stride=2,
            filter_size=5,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.Conv2DLayer(
            self.network,
            num_filters=32,
            stride=2,
            filter_size=3,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
        )
        self.network = lasagne.layers.MaxPool2DLayer(self.network, pool_size=(2, 2))
        self.network = lasagne.layers.DenseLayer(
            self.network,
            num_units=32,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform(),
        )

	#self.network_normalize = lasagne.layers.batch_norm(self.network)
        #network2
        self.layer_clinical = lasagne.layers.InputLayer(shape=(None, clinical_dim), input_var=self.Clinical)
        self.layer_clinical_dense = lasagne.layers.DenseLayer(self.layer_clinical,
                                                            num_units=128,
                                                            nonlinearity=lasagne.nonlinearities.rectify,)
        self.layer_clinical_dense3 = lasagne.layers.DenseLayer(self.layer_clinical_dense, num_units=32, nonlinearity=lasagne.nonlinearities.linear,)
	#self.clinical_normalize = lasagne.layers.batch_norm(self.layer_clinical_dense3)
        self.layer_combine = lasagne.layers.ConcatLayer([self.network, self.layer_clinical_dense3], axis=1)

        if standardize:
            self.network = lasagne.layers.standardize(self.network, self.offset,
                                                self.scale,
                                                shared_axes=0)
        self.standardize = standardize

        self.layer_combine1 = lasagne.layers.DenseLayer(self.layer_combine, num_units=32,
										    nonlinearity=lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())

        #self.layer_combine1drop = lasagne.layers.DropoutLayer(self.layer_combine1, p=0.5)

        self.layer_combine2 = lasagne.layers.DenseLayer(self.layer_combine1, num_units=8,
                                                         nonlinearity=lasagne.nonlinearities.linear,
                                                         W=lasagne.init.GlorotUniform())
        #self.layer_combine2drop = lasagne.layers.DropoutLayer(self.layer_combine2, p=0.5)

        self.multiview_output = lasagne.layers.DenseLayer(self.layer_combine2, num_units=1,
                                                          nonlinearity=lasagne.nonlinearities.linear,
                                                          W=lasagne.init.GlorotUniform())


#        self.params = lasagne.layers.get_all_params(self.multiview_output,
#                                                   trainable = True)

        self.params = [self.layer_combine1.W, self.layer_combine1.b,
		       self.layer_combine2.W, self.layer_combine2.b,
                       self.multiview_output.W, self.multiview_output.b]

        print "image_pretrain_name: ", image_pretrain_name
        print "clinical_pretrain_name: ", clinical_pretrain_name

        with np.load(image_pretrain_name) as f:
            img_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, img_param_values)
        with np.load(clinical_pretrain_name) as f:
            clinical_param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(self.layer_clinical_dense3, clinical_param_values)
#	if(sum(np.isnan(img_param_values))):
#		print ""
#		print "There is(are) NaN(s) in the image pretrained parameters!"
#	if(sum(np.isnan(clinical_param_values))):
#		pr:int ""
#		print "There is(are) NaN(s) in the clinical pretrained parameters!"
        print "finish loading model parameters..."

        # Relevant Functions
        self.partial_hazard = T.exp(self.risk(deterministic = True))

        # Set Hyper-parameters:
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.channel = channel
        self.width = width
        self.height = height
        self.clinical_dim = clinical_dim

    def _negative_log_likelihood(self, E, deterministic = False):
        risk = self.risk(deterministic)
        hazard_ratio = T.exp(risk)
        log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
        uncensored_likelihood = risk.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -T.sum(censored_likelihood)
        return neg_likelihood

    def _get_loss_updates(self,
    L1_reg = 0.0, L2_reg = 0.001,
    update_fn = lasagne.updates.nesterov_momentum,
    max_norm = None, deterministic = False,
    **kwargs):
        loss = (
            self._negative_log_likelihood(self.E, deterministic)
            + regularize_layer_params(self.multiview_output,l1) * L1_reg
            + regularize_layer_params(self.multiview_output, l2) * L2_reg
        )

        if max_norm:
            grads = T.grad(loss,self.params)
            scaled_grads = lasagne.updates.total_norm_constraint(grads, max_norm)
            updates = update_fn(
                grads, self.params, **kwargs
            )
            return loss, updates
        updates = update_fn(
                loss, self.params, **kwargs
            )

        return loss, updates

    def _get_train_valid_fn(self,
    L1_reg, L2_reg, learning_rate,
    **kwargs):
        loss, updates = self._get_loss_updates(
            L1_reg, L2_reg, deterministic = False,
            learning_rate=learning_rate, **kwargs
        )
        train_fn = theano.function(
            inputs = [self.l_in.input_var, self.layer_clinical.input_var, self.E],
            outputs = loss,
            updates = updates,
            name = 'train',
            on_unused_input = 'ignore'
        )

        valid_loss, _ = self._get_loss_updates(
            L1_reg, L2_reg, deterministic = True,
            learning_rate=learning_rate, **kwargs
        )

        valid_fn = theano.function(
            inputs = [self.l_in.input_var, self.layer_clinical.input_var, self.E],
            outputs = valid_loss,
            name = 'valid',
            on_unused_input = 'ignore'
        )
        return train_fn, valid_fn

    def get_concordance_index(self, x, t, e):
        compute_hazards = theano.function(
            inputs = [self.X],
            outputs = -self.partial_hazard,
            on_unused_input = 'ignore'
        )
        partial_hazards = compute_hazards(x)
        return concordance_index(t,
            partial_hazards,
            e)

    def get_partial_hazards(self, x, clinical):
        compute_hazards = theano.function(inputs = [self.l_in.input_var, self.layer_clinical.input_var], outputs = -self.partial_hazard, on_unused_input='ignore')
        partial_hazards = compute_hazards(x,clinical)
        return partial_hazards
    
    def get_concordance_index(self, imgs, clinical, t, e, index):
        partial_hazards = []
        for i in index :
            x = np.load(imgs[i])
            x = x.astype(theano.config.floatX)/255.0
            x = x.reshape(-1, self.channel, self.width, self.height)
            clin = clinical[i].astype(theano.config.floatX)
            clin = clin.reshape(-1,self.clinical_dim)
            partial_hazards.append(self.get_partial_hazards(x, clin))
        partial_hazards = np.asarray(partial_hazards)
        ci = concordance_index(np.asarray(t[index]), partial_hazards.reshape(np.asarray(t[index]).shape), np.asarray(e[index]))
        np.savetxt('risks.csv',np.c_[np.reshape(partial_hazards, -1), t[index], e[index]], delimiter=',', header="risk, time, status", comments='')
        return np.reshape(partial_hazards, -1), ci

    def train(self,data_path,clinical_path, label_path, train_index, test_index, valid_index,
    	    num_epochs = 5, batch_size = 10,
    		verbose = True, ratio = 0.8,
    		update_fn = lasagne.updates.nesterov_momentum,
    		**kwargs):
        if verbose:
            print('Start training DeepMultiSurv')
        label = pd.read_csv(label_path)
        clinical = pd.read_csv(clinical_path).convert_objects(convert_numeric=True).astype(np.float32)
        t= label["surv"].convert_objects(convert_numeric=True).astype(np.float32)
        e = label["status"].convert_objects(convert_numeric=True).astype(np.int32)
    	#t = label["surv"].to_numeric().astype(np.float32)
    	#e = label["status"].to_numeric().astype(np.int32)
        t = t.astype("float32").as_matrix()
        e = e.astype("int32").as_matrix()
        clinical = clinical.astype("float32").as_matrix()
        imgs = (data_path + label["img"].values).tolist()
        t_train = t[train_index]
        imgname = []
        for i in range(len(imgs)):
            imgname.append(imgs[i].split('.')[0]+'.'+imgs[i].split('.')[1]+".npy")
            #imgname.append(imgs[i].split('.')[0] + ".npy")
        imgs = imgname
        lr = theano.shared(np.array(self.learning_rate,
    	                                   dtype=np.float32))
        momentum = np.array(0, dtype=np.float32)
        train_fn, valid_fn = self._get_train_valid_fn(
    	        L1_reg=self.L1_reg, L2_reg=self.L2_reg,
    	        learning_rate=lr,
    	        momentum=momentum,
    	        update_fn=update_fn, **kwargs
    	    	)
    	for epoch_num in range(num_epochs):
            start_time = time.time()
    	    # iterate over training mini batches and update the weights
            lr = self.learning_rate / (1 + epoch_num * self.lr_decay)
            num_batches_train = int(np.ceil(len(t_train) / batch_size))
            train_losses = []
            # Power-Learning Rate Decay
            if self.momentum and epoch_num >= 10:
                momentum = self.momentum
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num, batch_size * (batch_num + 1))
                batch_index = train_index[batch_slice]
                img_batch = [imgs[i] for i in batch_index]
                x_batch = []
                for img in img_batch:
                        x_batch.append(np.load(img))
                x_batch = np.asarray(x_batch)
                x_batch = x_batch.astype(theano.config.floatX)/255.0
                x_batch = x_batch.reshape(-1, self.channel, self.width, self.height)
                e_batch = e[batch_index]
                t_batch = t[batch_index]
                c_batch = clinical[batch_index]
                c_batch = c_batch.astype(theano.config.floatX)
                    # Sort Training Data for Accurate Likelihood
                sort_idx = np.argsort(t_batch)[::-1]
                x_batch = x_batch[sort_idx]
                e_batch = e_batch[sort_idx]
                t_batch = t_batch[sort_idx]
                c_batch = c_batch[sort_idx]
                    # Initialize Metrics
                    	# best_validation_loss = np.inf
                    	# best_params = None
                    	# best_params_idx = -1
                if np.isnan(x_batch).any():
                    print "The images are with NAN value! Index: ", batch_index
                else:
                    loss = train_fn(x_batch, c_batch, e_batch)
                    train_losses.append(loss)
            train_loss = np.mean(train_losses)
            #ci_valid = self.get_concordance_index(imgs, clinical, t, e, valid_index)
            total_time = time.time() - start_time
            print("Epoch: %d, valid_ci=%f, train_loss=%f,  time=%fs"
    	         % (epoch_num + 1,0.0, train_loss, total_time))
        risk_test, ci_test = self.get_concordance_index(imgs,clinical, t,e,test_index)
        print "test: ", ci_test
        return risk_test, ci_test, t[test_index], e[test_index]

    def load_model(self, params):
        """
        Loads the network's parameters from a previously saved state.

        Parameters:
            params: a list of parameters in same order as network.params
        """
        lasagne.layers.set_all_param_values(self.network, params, trainable=True)

    def risk(self,deterministic = False):
        """
        Returns a theano expression for the output of network which is an
            observation's predicted risk.

        Parameters:
            deterministic: True or False. Determines if the output of the network
                is calculated determinsitically.

        Returns:
            risk: a theano expression representing a predicted risk h(x)
        """
        return lasagne.layers.get_output(self.multiview_output,
                                        deterministic = deterministic)

    def predict_risk(self, x):
        """
        Calculates the predicted risk for an array of observations.

        Parameters:
            x: (n,d) numpy array of observations.

        Returns:
            risks: (n) array of predicted risks
        """
        risk_fxn = theano.function(
            inputs = [self.X],
            outputs = self.risk(deterministic= True),
            name = 'predicted risk',
            on_unused_input='ignore'
        )
        return risk_fxn(x)

    def recommend_treatment(self, x, trt_i, trt_j, trt_idx = -1):
        """
        Computes recommendation function rec_ij(x) for two treatments i and j.
            rec_ij(x) is the log of the hazards ratio of x in treatment i vs.
            treatment j.

        .. math::

            rec_{ij}(x) = log(e^h_i(x) / e^h_j(x)) = h_i(x) - h_j(x)

        Parameters:
            x: (n, d) numpy array of observations
            trt_i: treatment i value
            trt_j: treatment j value
            trt_idx: the index of x representing the treatment group column

        Returns:
            rec_ij: recommendation
        """
        # Copy x to prevent overwritting data
        x_trt = np.copy(x)

        # Calculate risk of observations treatment i
        x_trt[:,trt_idx] = trt_i
        h_i = self.predict_risk(x_trt)
        # Risk of observations in treatment j
        x_trt[:,trt_idx] = trt_j;
        h_j = self.predict_risk(x_trt)

        rec_ij = h_i - h_j
        return rec_ij
        
    def plot_risk_surface(self, data, i = 0, j = 1,
        figsize = (6,4), x_lims = None, y_lims = None, c_lims = None):
        """
        Plots the predicted risk surface of the network with respect to two
        observed covarites i and j.

        Parameters:
            data: (n,d) numpy array of observations of which to predict risk.
            i: index of data to plot as axis 1
            j: index of data to plot as axis 2
            figsize: size of figure for matplotlib
            x_lims: Optional. If provided, override default x_lims (min(x_i), max(x_i))
            y_lims: Optional. If provided, override default y_lims (min(x_j), max(x_j))
            c_lims: Optional. If provided, override default color limits.

        Returns:
            fig: matplotlib figure object.
        """
        fig = plt.figure(figsize=figsize)
        X = data[:,i]
        Y = data[:,j]
        Z = self.predict_risk(data)

        if not x_lims is None:
            x_lims = [np.round(np.min(X)), np.round(np.max(X))]
        if not y_lims is None:
            y_lims = [np.round(np.min(Y)), np.round(np.max(Y))]
        if not c_lims is None:
            c_lims = [np.round(np.min(Z)), np.round(np.max(Z))]

        ax = plt.scatter(X,Y, c = Z, edgecolors = 'none', marker = '.')
        ax.set_clim(*c_lims)
        plt.colorbar()
        plt.xlim(*x_lims)
        plt.ylim(*y_lims)
        plt.xlabel('$x_{%d}$' % i, fontsize=18)
        plt.ylabel('$x_{%d}$' % j, fontsize=18)

        return fig
