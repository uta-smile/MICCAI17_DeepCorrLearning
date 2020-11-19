#!/usr/bin/python
# this is to implement a DeepCCASurv with Pre-trained model

import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import pandas as pd

class DeepMultiSurv:
    def __init__(self, learning_rate, channel, width, height, image_pretrain_name, clinical_pretrain_name, clinical_dim=5,
                 lr_decay=0.01, momentum=0.9,
                 L2_reg=0.0, L1_reg=0.0,
                 standardize=False
                 ):
        self.X = T.ftensor4('x')  # patients covariates
        self.E = T.ivector('e')  # the observations vector
        self.Clinical = T.matrix('c')

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

        # self.network = lasagne.layers.DropoutLayer(self.network, p=0.5)

        self.img_params_cca = lasagne.layers.get_all_params(self.network, trainable=True)

        # network 2
        self.layer_clinical = lasagne.layers.InputLayer(shape=(None, clinical_dim), input_var=self.Clinical)
        self.layer_clinical_dense = lasagne.layers.DenseLayer(self.layer_clinical,
                                                              num_units=128,
                                                              nonlinearity=lasagne.nonlinearities.rectify,
                                                              W=lasagne.init.GlorotUniform(), )
        self.layer_clinical_dense3 = lasagne.layers.DenseLayer(self.layer_clinical_dense, num_units=32,
                                                               nonlinearity=lasagne.nonlinearities.linear,
                                                               W=lasagne.init.GlorotUniform(), )

        # self.layer_clinical_dense3 = lasagne.layers.DropoutLayer(self.layer_clinical_dense3, p=0.5)

        self.clinical_params_cca = lasagne.layers.get_all_params(self.layer_clinical_dense3, trainable=True)

        with np.load(image_pretrain_name) as f:
            img_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, img_param_values)
        with np.load(clinical_pretrain_name) as f:
            clinical_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.layer_clinical_dense3, clinical_param_values)
        print "finish loading model parameters..."

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

    def _get_proj(self, model, deterministic=False):
        if model == 'image':
            return lasagne.layers.get_output(self.network, deterministic=deterministic)
        elif model == 'clinical':
            return lasagne.layers.get_output(self.layer_clinical_dense3, deterministic=deterministic)
        else:
            print "pls set model equals to either image or clinical"

    def _cca_loss(self, deterministic=False):
        img_proj = self._get_proj(model='image')
        clinical_proj = self._get_proj(model='clinical')
        img_mean = T.mean(img_proj, axis=0)
        img_centered = img_proj - img_mean
        clinical_mean = T.mean(clinical_proj, axis=0)
        clinical_centered = clinical_proj - clinical_mean
        corr_nr = T.sum(img_centered * clinical_centered, axis=0)
        corr_dr1 = T.sqrt(T.sum(img_centered * img_centered, axis=0) + 1e-8)
        corr_dr2 = T.sqrt(T.sum(clinical_centered * clinical_centered, axis=0) + 1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr / corr_dr
        cca_loss = T.sum(corr)
        return -1 * cca_loss

    def _get_loss_updates_cca(self, update_fn=lasagne.updates.nesterov_momentum, deterministic=False, **kwargs):
        img_loss = self._cca_loss(deterministic)
        clinical_loss = self._cca_loss(deterministic)
        img_updates = update_fn(img_loss, self.img_params_cca, **kwargs)
        clinical_updates = update_fn(clinical_loss, self.clinical_params_cca, **kwargs)
        # loss = self._cca_loss(deterministic)
        # img_updates = update_fn(loss, self.img_params_cca, **kwargs)
        # clinical_updates = update_fn(loss, self.clinical_params_cca, **kwargs)

        return img_loss, clinical_loss, img_updates, clinical_updates

    def _get_train_fn_cca(self, learning_rate, **kwargs):
        img_loss, clinical_loss, img_updates, clinical_updates = self._get_loss_updates_cca(learning_rate=learning_rate,
                                                                                            **kwargs)
        img_train_fn = theano.function(
            inputs=[self.X, self.Clinical],
            outputs=img_loss,
            updates=img_updates,
            name='imgtrain',
            on_unused_input='ignore'
        )
        clinical_train_fn = theano.function(
            inputs=[self.X, self.Clinical],
            outputs=clinical_loss,
            updates=clinical_updates,
            name='clinicaltrain',
            on_unused_input='ignore'
        )

        return img_train_fn, clinical_train_fn

    def train(self, data_path, clinical_path, label_path, train_index, test_index, valid_index, model_index,
              num_epochs=5, batch_size=1,
              verbose=True, ratio=0.8,
              update_fn=lasagne.updates.nesterov_momentum,
              **kwargs):
        if verbose:
            print('##########Start training DeepCCA#################')
        label = pd.read_csv(label_path)
        clinical = pd.read_csv(clinical_path).convert_objects(convert_numeric=True).astype(np.float32)
        t = label["surv"].convert_objects(convert_numeric=True).astype(np.float32)
        e = label["status"].convert_objects(convert_numeric=True).astype(np.int32)
        t = t.astype("float32").as_matrix()
        e = e.astype("int32").as_matrix()
        clinical = clinical.astype("float32").as_matrix()
        imgs = (data_path + label["img"].values).tolist()
        t_train = t[train_index]
        imgname = []
        for i in range(len(imgs)):
            imgname.append(imgs[i].split('.')[0]+"."+imgs[i].split('.')[1]+".npy") 
        imgs = imgname
        lr = theano.shared(np.array(self.learning_rate,
                                    dtype=np.float32))

        momentum = np.array(0, dtype=np.float32)
        img_train_fn, clinical_train_fn = self._get_train_fn_cca(learning_rate=lr, **kwargs)
        for epoch_num in range(num_epochs):
            start_time = time.time()
            lr = self.learning_rate / (1 + epoch_num * self.lr_decay)
            num_batches_train = int(np.ceil(len(t_train) / batch_size))
            img_train_losses = []
            clinical_train_losses = []
            if self.momentum and epoch_num >= 10:
                momentum = self.momentum
            for batch_num in range(num_batches_train):
                batch_slice = slice(batch_size * batch_num,
                                    batch_size * (batch_num + 1))
                batch_index = train_index[batch_slice]
                img_batch = [imgs[i] for i in batch_index]
                x_batch = []
                for img in img_batch:
                    x_batch.append(np.load(img))
                x_batch = np.asarray(x_batch)
                x_batch = x_batch.astype(theano.config.floatX) / 255.0
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

                if np.isnan(x_batch).any():
                    print "The images are with NAN value! Index: ", batch_index
                else:
                    #  train_loss = img_train_fn(x_batch, c_batch)
                    img_loss = img_train_fn(x_batch, c_batch)
                    clinical_loss = clinical_train_fn(x_batch, c_batch)
                    img_train_losses.append(img_loss)
                    clinical_train_losses.append(clinical_loss)

            img_train_loss = np.mean(img_train_losses)
            clinical_train_loss = np.mean(clinical_train_losses)
            total_time = time.time() - start_time
            print("Epoch: %d, img_train_loss=%f, clinical_train_loss=%f, time=%fs"
                  % (epoch_num + 1, img_train_loss, clinical_train_loss, total_time))
        imgmodel_name = 'imgmodel%d.npz' % model_index
        clinicalmodel_name = 'moleculemodel%d.npz' % model_index
        np.savez(imgmodel_name, *lasagne.layers.get_all_param_values(self.network))
        np.savez(clinicalmodel_name, *lasagne.layers.get_all_param_values(self.layer_clinical_dense3))

    def load_model(self, params):
        lasagne.layers.set_all_param_values(self.network, params, trainable=True)

    def risk(self, deterministic=False):
        return lasagne.layers.get_output(self.network,
                                         deterministic=deterministic)
