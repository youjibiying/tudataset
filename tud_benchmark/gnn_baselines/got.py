import torch
import torch.nn as nn



class Sinkhron(torch.nn.Module):
    def __init__(self, bs=None, beta=0.5, got_beta=0.1, is_uniform=True, got_iteration=20, got_wd_iteration=20,
                 wd_iteration=50,
                 mask=None, args=None):
        super(Sinkhron, self).__init__()
        # self.C = C
        self.beta = beta
        self.got_beta = got_beta
        self.mask = mask
        self.bs = bs if args is None else args.batch_size
        self.N_s = None
        self.N_t = None
        self.is_uniform = is_uniform
        self.wd_iteration = wd_iteration
        self.got_iteration = got_iteration
        self.got_wd_iteration = got_wd_iteration

        self.args = args

    def marginal_prob_unform(self, N_s=None, N_t=None, mask=None, ):
        if mask is not None:
            mask = mask.float()
            mask_mean = (1 / mask.sum(1)).unsqueeze(1)
            mu = mask * mask_mean
            mu = mu.unsqueeze(2)
        else:
            mu = torch.ones(self.bs, N_s, 1) / N_s
        mv = mu.clone().detach()
        return mu, mv

    def init_A_T(self, C, N_s, N_t, beta, mask):
        mu, mv = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)  # clone, 不共享数据。detach，不回传梯度
        if mask is not None:
            # mask = mask0.float().unsqueeze(2).clone()
            # mask.requires_grad = True
            # N_s = mask.sum(1).unsqueeze(1).type_as(C)
            mask = mask.float()
            # mask_mean = (1 / mask.sum(1)).unsqueeze(1)
            # sigma = mask * mask_mean
            # sigma = sigma.unsqueeze(2)
            # sigma = torch.ones(self.bs, C.shape[-2], 1).to(self.device).div(N_s)
            # self.T = torch.ones(bs, n, m)
            # todo I I^T with mask
            mask = mask.unsqueeze(2).clone()
            T = torch.bmm(mask, (mask.transpose(2, 1)))  # torch.ones_like(C)
            # A = torch.exp(-C / self.beta).float()
        else:
            # N_s = C.shape[-2] if self.N_s is not None else self.N_s
            # N_t = C.shape[-1] if self.N_t is not None else self.N_t
            # sigma = torch.ones(self.bs, N_s, 1) / N_s
            T = torch.ones(self.bs, N_s, N_t)
        A = torch.exp(-C / beta).float()
        sigma = mu.clone().detach()

        return A, T, mu, mv, sigma

    def IPOT_batch_uniform_mask_T(self, C, bs, N_s, N_t, beta=0.5, iteration=50, mask=None):
        '''
        Sinkhorn algorithm, with unifiorm distribution
        Args:
            C: cost matirx:distance matrix  bs*n*m
            bs: batch_size
            n: source dim
            m: target dim
            beta: A=exp(-C/beta)
            iteration:
            mask: bs*n
        Returns:
                T: transition matrix
        '''
        # c: bs by n by m

        # if self.mask is not None:
        # sigma = torch.ones(bs, int(m), 1).cuda() / float(m)
        # T = torch.ones(bs, n, m).cuda()
        # A = torch.exp(-C / beta).float().cuda()
        # T = self.T
        # sigma = self.sigma
        # if self.is_uniform:
        self.N_s = N_s
        self.N_t = N_t
        A, T, mu, mv, sigma = self.init_A_T(C, N_s=N_s, N_t=N_t, beta=beta, mask=mask)
        for t in range(self.wd_iteration):
            Q = A * T  # bs * n * m
            T1=T
            for k in range(1):
                b = torch.bmm(Q, sigma)
                b.masked_fill_(mask=~mask.unsqueeze(2), value=1e-8)
                # delta = 1 / (n * torch.bmm(Q, self.sigma))

                delta = mu / b
                a = torch.bmm(torch.transpose(Q, 1, 2), delta)  # Q^T delta
                a.masked_fill_(mask=~mask.unsqueeze(2), value=1e-8)
                sigma = mv / a
                # self.sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2, 1)  # diag(delta)Q\diag(\simga) 左行右列
            err = (T - T1).abs().sum(-1).max()

            # actual_nits += 1
            if err.item() < 0.1:
                break
        return T  # .detach()

    def forward(self, C, bs, N_s, N_t, mask):
        '''
        gat the optimal transport distance with Sinkhorm
        Args:
            C: cost matrix
            bs:
            n:
            m:
            iteration:

        Returns:

        '''
        self.device = C.device
        # C = C.float().cuda()
        # Sinkhorm
        T = self.IPOT_batch_uniform_mask_T(C=C, bs=bs, N_s=N_s, N_t=N_t, beta=self.beta, mask=mask)
        # T= torch.ones_like(C)
        distance=torch.sum(T * C, dim=(-2, -1))
        # temp = torch.bmm(torch.transpose(C, 1, 2), T)  # (C^T)T
        # distance = self.batch_trace(temp, N_t, bs)  # 取出迹， 作为矩阵的内积tr(Q^TT)=<Q,T>
        return distance, T, C

    def GW_torch_batch(self, Cs, Ct, bs, N_s, N_t, p, q, mask=None):
        '''
        Computing Gromov-wasserstein disatnce with sinkhorn
        Args:
            Cs: source pair-distance matrix
            Ct: target pair-distance matrix
            bs:
            n:
            m:
            p: probability vectors ,marginal distribution on source domain
            q: probability vectors ,marginal distribution on target domain
            beta:
            iteration:
            OT_iteration:

        Returns:
            T, pseudo-cost matrix Cgamma C_{st} - 2C_s T C_t^T
        '''
        one_m = torch.ones(bs, N_t, 1).float().to(mask.device) * (mask.unsqueeze(2))
        one_n = torch.ones(bs, N_s, 1).float().to(mask.device) * mask.unsqueeze(2)
        # compute cross-domain similarities C_{st} = C_x^2 p I_m^T + C_tq(C_t^2)^T
        Cst = torch.bmm(torch.bmm(Cs ** 2, p) * (mask.unsqueeze(2)), torch.transpose(one_m, 1, 2)) + \
              torch.bmm(one_n, torch.bmm(torch.transpose(q, 1, 2), torch.transpose(Ct ** 2, 1, 2)))  # bs by n by m
        gamma = torch.bmm(p, q.transpose(2, 1))  # outer product pq^T, init T (transition matrix)
        # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
        for i in range(self.got_iteration):
            # pseudo-cost matrix
            C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
            # shared T
            # C_gamma = 1.0 *
            # Sinkhorn iteration to get transition matrix T
            gamma = self.IPOT_batch_uniform_mask_T(C=C_gamma, bs=bs, N_s=N_s, N_t=N_t, beta=self.got_beta,
                                                   mask=mask)  # (C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def GW_distance(self, X, Y, p, q, beta, b_A=None, mask=None):
        '''
        :param X, Y: Source and target embeddings , batchsize by embed_dim by n
        :param p, q: probability vectors
        :param lamda: regularization which uses in sinkhorn
        :return: GW distance: L= C_{st} - 2C_s T C_t^T, T=sinkhorm(L), D_{GW}=<L,T>=tr(L^T T)
        '''
        Cs = self.cos_batch_torch(X, X, mask=mask).float()  # .cuda()
        Ct = self.cos_batch_torch(Y, Y, mask=mask).float()  # .cuda()
        # pdb.set_trace()
        if b_A is not None:
            Cs = Cs * b_A
            Ct = Ct * b_A
        else:
            self.to_adjancy(dist=Cs, beta=0.1)
            self.to_adjancy(dist=Ct, beta=0.1)
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        # L= C_{st} - 2C_s T C_t^T, T=sinkhorm(L)
        T, Cst = self.GW_torch_batch(Cs, Ct, bs, n, m, p, q, mask=mask)
        # inner product with matrix
        temp = torch.bmm(torch.transpose(Cst, 1, 2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_distance_uniform(self, X, Y, bs, beta=1e-1, b_A=None, mask=None):
        '''
        sourse domain with distance matrix X ,and target with Y, to compute the pair transport between two domains, with marginal probability uniform
        Args:
            X: distance matrix in source domain
            Y: distance matrix in target domain
            lamda:
            iteration: GW iterate
            OT_iteration: sinknorm iterate

        Returns:

        '''
        self.device = X.device
        N_s = X.size(2)
        N_t = Y.size(2)
        # bs = X.size(0)
        # p = (torch.ones(bs, m, 1)/m).cuda()
        # q = (torch.ones(bs, n, 1)/n).cuda()
        p, q = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)  # clone, 不共享数据。detach，不回传梯度

        return self.GW_distance(X, Y, p, q, beta=beta, b_A=b_A, mask=mask)

    def cos_batch_torch(self, x, y, mask=None):
        "Returns the cosine distance batchwise"
        # x is the image feature: bs * d * m * m
        # y is the audio feature: bs * d * nF
        # return: bs * n * m
        # print(x.size())
        bs = x.size(0)
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m^2
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # return cos_dis.transpose(2,1)
        # TODO:

        # res = torch.nn.ReLU()
        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            cos_dis = cos_dis * mask_matrix
        return cos_dis.transpose(2, 1)

    def cost_matrix_batch_torch(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the source feature: bs * d * m
        # y is the target feature: bs * d * m
        # return: bs * n * m
        # print(x.size())
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # cos_dis = - cos_dis
        return cos_dis.transpose(2, 1)

    # def norm_distance(self,X,Y,p):
    #     for x,y in zip(X,Y):
    #         C = torch.norm(X[:,None]-Y, p=2, dim=2)
    #     return C

    def euclidean_cost_matrix(self, x, y, p=2, mask=None):
        '''

        Args:
            x: batch*N*d
            y: batch*N*d
            p: L_p norm
            mask:  batch*N

        Returns:
            Returns the matrix of $|x_i-y_j|^p$.
        '''
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)

        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)

        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            C = C * mask_matrix
        return C

    def pairwise_distances(self, x, y=None, p=2):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        bs = x.shape[0]
        x_norm = (x ** 2).sum(-1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, -2, -1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, -2, -1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    # def euclidean_dist(self, x, y):
    #     m, n = x.size(1), y.size(1)
    #     xx = torch.pow(x, 2).sum(2, keepdim=True).expand(m, n)
    #     yy = torch.pow(y, 2).sum(2, keepdim=True).expand(n, m).t()
    #     dist = xx + yy
    #     dist.addmm_(1, -2, x, y.t())
    #     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #     return dist

    def to_adjancy(self, dist, beta):
        beta = 0.1
        min_score = dist.min()
        max_score = dist.max()
        threshold = min_score + beta * (max_score - min_score)
        res = dist - threshold
        return torch.nn.functional.relu(res)

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py
class LogSinkhorn(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=0.1, thresh=0.1, max_iter=100, reduction='none'):
        super(LogSinkhorn, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh
        self.mask_matrix = None

    def marginal_prob_unform(self, N_s=None, N_t=None, mask=None, ):
        if mask is not None:
            mask = mask.float()
            # uniform distribution
            mask_mean = (1 / mask.sum(1)).unsqueeze(1)
            mu = mask * mask_mean  # 1/n
            # mu = mu.unsqueeze(2)
        else:
            mu = torch.ones(self.bs, N_s) / N_s
        nu = mu.clone().detach()
        return mu, nu

    def forward(self, x, y, C=None, A=None, mask=None):
        # The Sinkhorn algorithm takes as input three variables :
        if C is None:
            C = self._cost_matrix(x, y)  # Wasserstein cost function
            C = C / C.max()
        if A is not None:
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C
        N_s = x.shape[-2]
        N_t = y.shape[-2]
        if x.dim() == 2:
            self.bs = 1
        else:
            self.bs = x.shape[0]

        # both marginals are fixed with equal weights
        if mask is None:
            mu = torch.empty(self.bs, N_s, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_s).squeeze()
            nu = torch.empty(self.bs, N_t, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_t).squeeze()
        else:
            mu, nu = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = self.thresh

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update

            # u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v, A=A), dim=-1)) + u
            # v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            # todo in fact, just use the withdout mask is right, because log(1e-8)-log(1e-8)=0
            if mask is None:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            else:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                u = mask * u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
                v = mask * v

            # err = (u - u1).abs().sum(-1).mean()
            err = (u - u1).abs().sum(-1).max()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        # pi = torch.exp(self.M(C, U, V))
        # Transport plan pi = diag(a)*K*A*diag(b)
        # pi = torch.exp(self.M(C, U, V, A=A)) + torch.exp(-C / self.eps) * (1 - A)
        pi = self.exp_M(C, U, V, A=A)
        # if A is not None:
        #     pi = torch.exp(self.M(C, U, V)) * A
        # else:
        #     pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        # print(pi[0], pi[0].sum(1))
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        if torch.isnan(cost.sum()):
            print(pi)
            raise
        return cost, pi, C

    def M(self, C, u, v, A=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        # if A is not None:
        #     S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps + torch.log(A + 1e-8)
        #     return S

        S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return S

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-a).to(torch.bool),value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-A).to(torch.bool),value=0)

            return S
        elif self.mask_matrix is not None:
            return self.mask_matrix * torch.exp(self.M(C, u, v))
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1, mask=None):
        s = torch.sum(input_tensor, dim=dim)
        out = torch.log(1e-8 + s)
        if torch.isnan(out.sum()):
            raise
        if mask is not None:
            out = mask * out
        return out

    def cost_matrix_batch_torch(self, x, y, mask=None, add_eye_diag=False):
        "Returns the cosine distance batchwise"
        # x is the source feature: bs * d * m
        # y is the target feature: bs * d * m
        # return: bs * n * m
        # print(x.size())
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        similarity = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - similarity  # to minimize this value
        # cos_dis = - cos_dis
        if add_eye_diag:
            cos_dis = cos_dis + torch.eye(cos_dis.shape[-1]).repeat(cos_dis.shape[0],1,1).to(cos_dis.device)
        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            self.mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            cos_dis = cos_dis * self.mask_matrix
            similarity=similarity*self.mask_matrix
        if torch.isnan(cos_dis.sum()):
            raise
        return cos_dis.transpose(2, 1), similarity

    def cost_matrix_torch(self, x, y):
        "Returns the cosine distance"
        # x is the image embedding
        # y is the text embedding
        D = x.size(0)
        x = x.view(D, -1)
        assert (x.size(0) == y.size(0))
        x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) + 1e-12)
        cos_dis = torch.mm(torch.transpose(y, 0, 1), x)  # .t()
        cos_dis = 1 - cos_dis  # to minimize this value
        return cos_dis

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

