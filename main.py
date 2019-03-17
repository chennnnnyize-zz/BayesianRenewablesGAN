#!/usr/bin/env python
import os
import sys
import argparse
import json
import time

import numpy as np
from math import ceil
from PIL import Image

import tensorflow as tf
from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset
from bgan_orig import BDCGAN
from numpy import shape
from bgan_util import load_wind, load_mixture, noise_mixture
import csv
import matplotlib.pyplot as plt

import time


def get_session():
    if tf.get_default_session() is None:
        print "Creating new session"
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print "Using old session"
        _SESSION = tf.get_default_session()

    return _SESSION



def b_dcgan(args):
    z_dim = 100
    #x_dim = dataset.x_dim
    batch_size = 64
    n_epochs=50
    #dataset_size = dataset.dataset_size

    x_dim, dataset_size, data = load_wind()
    print("dataset_size", dataset_size)
    print("x_dim", x_dim)

    session = get_session()

    tf.set_random_seed(2222)

    dcgan = BDCGAN(x_dim, z_dim, dataset_size, batch_size=batch_size, J=args.J, M=args.M,
                   lr=args.lr, optimizer=args.optimizer, gen_observed=args.gen_observed,
                   num_classes=1)

    print "Starting session"
    session.run(tf.global_variables_initializer())

    print "Starting training loop"


    base_learning_rate = args.lr  # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay
    train_iter=0

    for epoch in range(n_epochs):
        print("epoch" + str(epoch))
        index = np.arange(len(data))
        np.random.shuffle(index)
        trX = data[index]

        for start, end in zip(
                range(0, len(trX), batch_size),
                range(batch_size, len(trX), batch_size)
        ):


            optimizer_dict = {"adv_d": dcgan.d_optim,
                                  "gen": dcgan.g_optims}

            learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                        min(1.0, (train_iter * batch_size) / float(dataset_size)))

            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
            Xs = trX[start:end].reshape([-1, 24, 24, 1])

            # regular GAN
            _, d_loss = session.run([optimizer_dict["adv_d"], dcgan.d_loss], feed_dict={dcgan.inputs: Xs,
                                                                                        dcgan.z: batch_z,
                                                                                        dcgan.d_learning_rate: learning_rate})
            #print(Xs[0])
            if args.wasserstein:
                session.run(dcgan.clip_d, feed_dict={})

            g_losses = []
            for gi in xrange(dcgan.num_gen):
                # compute g_sample loss
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
                for m in range(dcgan.num_mcmc):
                    _, g_loss = session.run([optimizer_dict["gen"][gi * dcgan.num_mcmc + m],
                                             dcgan.generation["g_losses"][gi * dcgan.num_mcmc + m]],
                                            feed_dict={dcgan.z: batch_z, dcgan.g_learning_rate: learning_rate})
                    g_losses.append(g_loss)

            #Print loss and save wind samples
            if train_iter > 3000 and train_iter % 500 == 0:

                print "Iter %i" % train_iter
                # collect samples
                all_sampled_imgs = []
                for gi in xrange(dcgan.num_gen):
                    _imgs, _ps = [], []
                    for _ in range(3):
                        sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                        sampled_imgs, sampled_probs = session.run(
                            [dcgan.generation["gen_samplers"][gi * dcgan.num_mcmc],
                             dcgan.generation["d_probs"][gi * dcgan.num_mcmc]],
                            feed_dict={dcgan.z: sample_z})
                        _imgs.append(sampled_imgs)
                        _ps.append(sampled_probs)

                    sampled_imgs = np.concatenate(_imgs);
                    sampled_probs = np.concatenate(_ps)
                    #all_sampled_imgs.append([sampled_imgs, sampled_probs[:, 1:].sum(1)])

                    generated_samples = sampled_imgs.reshape(-1, 576)
                    generated_samples = (generated_samples+1.)/2. * 16

                    csvfile = file('Results/samples_%s_%i.csv' % (train_iter,gi), 'wb')
                    writer = csv.writer(csvfile)
                    writer.writerows(generated_samples)

                print "Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % gl for gl in g_losses]))

            train_iter+=1

            '''if args.save_weights:
                var_dict = {}
                for var in tf.trainable_variables():
                    var_dict[var.name] = session.run(var.name)

                np.savez_compressed(os.path.join(args.out_dir,
                                                 "weights_%i.npz" % train_iter),
                                    **var_dict)'''



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        default="Results",
                        help="location of outputs (root location, which exists)")

    parser.add_argument('--gen_observed',
                        type=int,
                        default=1000,
                        help='number of data "observed" by generator')

    parser.add_argument('--prior_std',
                        type=float,
                        default=0.5, #default=1.0
                        help="NN weight prior std.")

    #Number of generators
    parser.add_argument('--numz',
                        type=int,
                        dest="J",
                        default=8,
                        help="number of samples of z to integrate it out")

    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=4,
                        help="number of MCMC NN weight samples per z")

    parser.add_argument('--wasserstein',
                        action="store_true",
                        help="wasserstein GAN")

    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate")

    parser.add_argument('--lr_decay',
                        type=float,
                        default=3.0,
                        help="learning rate")

    parser.add_argument('--optimizer',
                        type=str,
                        default="adam",
                        help="optimizer --- 'adam' or 'sgd'")

    args = parser.parse_args()


    #np.random.seed(2222)
    tf.set_random_seed(2222)

    if not os.path.exists(args.out_dir):
        print "Creating %s" % args.out_dir
        os.makedirs(args.out_dir)
    args.out_dir = os.path.join(args.out_dir, "Result")
    #os.makedirs(args.out_dir)

    import pprint
    with open(os.path.join(args.out_dir, "hypers.txt"), "w") as hf:
        hf.write("Hyper settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))


    ### main function starts here!
    b_dcgan( args)
