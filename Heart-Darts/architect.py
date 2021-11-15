import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        # We use Adam as the optimizer for α
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    # 不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新
    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """计算w'= w − ξ*dwLtrain(w, α) """
        loss = self.model._loss(input, target)  # Ltrain
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss,
                                             self.model.parameters())).data + self.network_weight_decay * theta  # 前面的是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))  # w − ξ*dwLtrain(w, α)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):

        self.optimizer.zero_grad()  # 清除上一步的残余更新参数值
        if unrolled:  # 用论文的提出的方法
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:  # 不用论文提出的bilevel optimization，只是简单的对α求导
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()  # 应用梯度：根据反向传播得到的梯度进行参数的更新， 这些parameters的梯度是由loss.backward()得到的，optimizer存了这些parameters的指针
        # 因为这个optimizer是针对alpha的优化器，所以他存的都是alpha的参数

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)"""
        # w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # Lval
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        # print("Loss in validation set:", unrolled_loss)
        # dLval
        unrolled_loss.backward()
        # dαLval(w',α)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # dw'Lval(w',α)
        vector = [v.grad.data for v in unrolled_model.parameters()]
        # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)   其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        # 对α进行更新
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    # 对应optimizer.step()，对新建的模型的参数进行更新
    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()  # epsilon
        # dαLtrain(w+,α)
        for p, v in zip(self.model.parameters(), vector):
            # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # dαLtrain(w-,α)
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
