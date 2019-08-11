import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.dataloader import DataLoader


import os
import sys
import logging
import random
import time

from dataset import ImageFolder
from networks import *
from utils import *
from initializer import KaimingUniform, BiasInitializer


def param_dicts_merge(*args):
    pdict = gluon.parameter.ParameterDict()
    for a in args:
        pdict.update(a)
    return pdict


def force_init(params, init):
    for _, v in params.items():
        v.initialize(init, None, init, force_reinit=True)


class UGATIT:
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.num_workers = args.num_workers

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.dev = mx.cpu() if self.device == 'cpu' else mx.gpu(args.gpu)
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.debug = args.debug

        if not self.benchmark_flag:
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

        print("##### Information #####")
        print("# dev : ", self.dev)
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print("\n##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print("\n##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print("\n##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    def build_model(self):
        # DataLoader
        train_transform = transforms.Compose([
            transforms.RandomFlipLeftRight(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomResizedCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join(
            'dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join(
            'dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join(
            'dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join(
            'dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(
            self.trainA, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.trainB_loader = DataLoader(
            self.trainB, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                      n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch,
                                      n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        self.whole_model = nn.HybridSequential()
        self.whole_model.add(
            *[self.genA2B, self.genB2A, self.disGA, self.disGB, self.disLA, self.disLB])

        self.whole_model.hybridize(static_alloc=False, static_shape=False)

        """ Define Loss """
        self.L1_loss = gloss.L1Loss()
        self.MSE_loss = gloss.L2Loss()
        self.BCE_loss = gloss.SigmoidBCELoss()

        """ Initialize Parameters"""
        params = self.whole_model.collect_params()
        block = self.whole_model
        if not self.debug:
            force_init(block.collect_params('.*?_weight'), KaimingUniform())
            force_init(block.collect_params('.*?_bias'), BiasInitializer(params))
            block.collect_params('.*?_rho').initialize()
            block.collect_params('.*?_gamma').initialize()
            block.collect_params('.*?_beta').initialize()
            block.collect_params('.*?_state_.*?').initialize()
        else:
            pass
        block.collect_params().reset_ctx(self.dev)

        """ Trainer """
        self.G_params = param_dicts_merge(
            self.genA2B.collect_params(),
            self.genB2A.collect_params(),
        )
        self.G_optim = gluon.Trainer(
            self.G_params,
            'adam',
            dict(learning_rate=self.lr, beta1=0.5,
                 beta2=0.999, wd=self.weight_decay),
        )
        self.D_params = param_dicts_merge(
            self.disGA.collect_params(),
            self.disGB.collect_params(),
            self.disLA.collect_params(),
            self.disLB.collect_params()
        )
        self.D_optim = gluon.Trainer(
            self.D_params,
            'adam',
            dict(learning_rate=self.lr, beta1=0.5,
                 beta2=0.999, wd=self.weight_decay),
        )

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        start_iter = 1

        if self.resume:
            self._load_impl(self.resume)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.set_learning_rate(
                    self.G_optim.learning_rate - self.lr / (self.iteration // 2))
                self.D_optim.set_learning_rate(
                    self.D_optim.learning_rate - self.lr / (self.iteration // 2))
                print('Set Learning rate: G: {.8f}, D: {.8f}'.format(
                    self.G_optim.learning_rate, self.D_optim.learning_rate))

            data_tic = time.time()
            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter)
            real_A = real_A.as_in_context(self.dev)
            real_B = real_B.as_in_context(self.dev)
            data_time = time.time() - data_tic

            # Update D
            self.D_params.zero_grad()
            with autograd.record():

                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                D_ad_loss_GA = self.MSE_loss(real_GA_logit, mx.nd.ones_like(
                    real_GA_logit)) + self.MSE_loss(fake_GA_logit, mx.nd.zeros_like(fake_GA_logit))
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, mx.nd.ones_like(
                    real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, mx.nd.zeros_like(fake_GA_cam_logit))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit, mx.nd.ones_like(
                    real_LA_logit)) + self.MSE_loss(fake_LA_logit, mx.nd.zeros_like(fake_LA_logit))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, mx.nd.ones_like(
                    real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, mx.nd.zeros_like(fake_LA_cam_logit))
                D_ad_loss_GB = self.MSE_loss(real_GB_logit, mx.nd.ones_like(
                    real_GB_logit)) + self.MSE_loss(fake_GB_logit, mx.nd.zeros_like(fake_GB_logit))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, mx.nd.ones_like(
                    real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, mx.nd.zeros_like(fake_GB_cam_logit))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit, mx.nd.ones_like(
                    real_LB_logit)) + self.MSE_loss(fake_LB_logit, mx.nd.zeros_like(fake_LB_logit))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, mx.nd.ones_like(
                    real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, mx.nd.zeros_like(fake_LB_cam_logit))

                D_loss_A = self.adv_weight * \
                    (D_ad_loss_GA + D_ad_cam_loss_GA +
                     D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * \
                    (D_ad_loss_GB + D_ad_cam_loss_GB +
                     D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
            self.D_optim.step(1)

            # Update G
            self.G_params.zero_grad()
            with autograd.record():
                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

                G_ad_loss_GA = self.MSE_loss(
                    fake_GA_logit, mx.nd.ones_like(fake_GA_logit))
                G_ad_cam_loss_GA = self.MSE_loss(
                    fake_GA_cam_logit, mx.nd.ones_like(fake_GA_cam_logit))
                G_ad_loss_LA = self.MSE_loss(
                    fake_LA_logit, mx.nd.ones_like(fake_LA_logit))
                G_ad_cam_loss_LA = self.MSE_loss(
                    fake_LA_cam_logit, mx.nd.ones_like(fake_LA_cam_logit))
                G_ad_loss_GB = self.MSE_loss(
                    fake_GB_logit, mx.nd.ones_like(fake_GB_logit))
                G_ad_cam_loss_GB = self.MSE_loss(
                    fake_GB_cam_logit, mx.nd.ones_like(fake_GB_cam_logit))
                G_ad_loss_LB = self.MSE_loss(
                    fake_LB_logit, mx.nd.ones_like(fake_LB_logit))
                G_ad_cam_loss_LB = self.MSE_loss(
                    fake_LB_cam_logit, mx.nd.ones_like(fake_LB_cam_logit))

                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, mx.nd.ones_like(
                    fake_B2A_cam_logit)) + self.BCE_loss(fake_A2A_cam_logit, mx.nd.zeros_like(fake_A2A_cam_logit))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, mx.nd.ones_like(
                    fake_A2B_cam_logit)) + self.BCE_loss(fake_B2B_cam_logit, mx.nd.zeros_like(fake_B2B_cam_logit))

                G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                    self.cycle_weight * G_recon_loss_A + self.identity_weight * \
                    G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                    self.cycle_weight * G_recon_loss_B + self.identity_weight * \
                    G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
            self.G_optim.step(1)

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            mx.nd.waitall()
            print("[%5d/%5d] time: %.4f data_time: %.4f, d_loss: %.8f, g_loss: %.8f" % (
                  step, self.iteration,
                  time.time() - start_time,
                  data_time,
                  Discriminator_loss.asscalar(),
                  Generator_loss.asscalar()))

            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                for _ in range(train_sample_num):
                    try:
                        real_A, _ = next(trainA_iter)
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = next(trainA_iter)

                    try:
                        real_B, _ = next(trainB_iter)
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _ = next(trainB_iter)

                    real_A = real_A.as_in_context(self.dev)
                    real_B = real_B.as_in_context(self.dev)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = next(testA_iter)
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = next(testA_iter)

                    try:
                        real_B, _ = next(testB_iter)
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = next(testB_iter)

                    real_A = real_A.as_in_context(self.dev)
                    real_B = real_B.as_in_context(self.dev)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(
                                                                   denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(
                                                                   fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                         'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                         'img', 'B2A_%07d.png' % step), B2A * 255.0)

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir,
                                       self.dataset, 'model'), step)

            if step % 1000 == 0:
                model_name = os.path.join(
                    self.result_dir, self.dataset + '_latest.params')
                self._save_impl(model_name)

    def _save_impl(self, model_name):
        self.whole_model.save_parameters(model_name)
        self.G_optim.save_states(model_name + '.G_optim')
        self.D_optim.save_states(model_name + '.D_optim')

    def _load_impl(self, model_name):
        self.whole_model.load_parameters(model_name, ctx=self.dev)
        self.G_optim.load_states(model_name + '.G_optim', ctx=self.dev)
        self.D_optim.load_states(model_name + '.D_optim', ctx=self.dev)

    def save(self, dir_name, step):
        model_name = os.path.join(
            dir_name, self.dataset + '_params_%07d.params' % step)
        self._save_impl(model_name)

    def load(self, dir_name, step):
        model_name = os.path.join(
            dir_name, self.dataset + '_params_%07d.params' % step)
        self._load_impl(model_name)

    def test(self):
        raise NotImplementedError()
        model_list = os.path.join(
            self.result_dir, self.dataset, 'model', '*.params')
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir,
                                   self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                       cam(tensor2numpy(
                                                           fake_A2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(
                                                           denorm(fake_A2A[0]))),
                                                       cam(tensor2numpy(
                                                           fake_A2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(
                                                           denorm(fake_A2B[0]))),
                                                       cam(tensor2numpy(
                                                           fake_A2B2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                     'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                       cam(tensor2numpy(
                                                           fake_B2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(
                                                           denorm(fake_B2B[0]))),
                                                       cam(tensor2numpy(
                                                           fake_B2A_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(
                                                           denorm(fake_B2A[0]))),
                                                       cam(tensor2numpy(
                                                           fake_B2A2B_heatmap[0]), self.img_size),
                                                       RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,
                                     'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
