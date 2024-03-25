# import torch
# import torch.nn as nn

# print(torch.cuda.is_available())

# class GaLore(nn.Module):
#     def __init__(self, model, rank=4, update_freq=200, galore_params=None):
#         super(GaLore, self).__init__()
#         self.model = model.to(torch.device("cuda"))  # Move model to GPU
#         print(f"Model device: {next(self.model.parameters()).device}")  # Check model's device
#         self.rank = rank
#         self.update_freq = update_freq
#         self.n_step = 0
#         if galore_params is not None:
#             self.params_list = galore_params
#         else:
#             self.params_list = self.model.named_parameters()
#         # self.params_list = [(name, param) for name, param in params_list_temp if param.requires_grad and len(param.shape) > 1]
#         self.P = {}
#         self.Q = {}

#         for name, param in self.params_list:
#             if param.requires_grad and len(param.shape) > 1:
#                 self.P[name] = torch.empty(
#                     (param.data.shape[0], self.rank),
#                     dtype=param.data.dtype,
#                     device=param.data.device,
#                 )
#                 self.Q[name] = torch.empty(
#                     (param.data.shape[1], self.rank),
#                     dtype=param.data.dtype,
#                     device=param.data.device,
#                 )
#                 nn.init.orthogonal_(self.P[name])
#                 nn.init.orthogonal_(self.Q[name])

#     def project(self, grad, name):
#         print(f"the grad is {grad}")
#         print(f"Shape of grad tensor: {grad.shape}")
#         return torch.matmul(self.P[name].t(), torch.matmul(grad, self.Q[name]))

#     def project_back(self, lor_grad, name):
#         return torch.matmul(self.P[name], torch.matmul(lor_grad, self.Q[name].t()))

#     def update_projections(self):
#         with torch.no_grad():
#             for name, param in self.params_list:
#                 if param.requires_grad:
#                     if len(param.shape) > 1:
#                         grad = param.grad
#                         U, _, Vt = torch.linalg.svd(grad, full_matrices=False)
#                         self.P[name] = U[:, : self.rank]
#                         self.Q[name] = Vt[: self.rank, :].t()

#     def step(self, update_func):
#         for name, param in self.params_list:
#             if param.requires_grad:
#                 if len(param.shape) > 1:
#                     grad = param.grad
#                     # grad.retain_grad()
#                     lor_grad = self.project(grad, name)
#                     lor_update = update_func(lor_grad)
#                     update = self.project_back(lor_update, name)
#                     param.data += update
#                 else:
#                     update_func(param.grad)

#         self.n_step += 1
#         if self.n_step % self.update_freq == 0:
#             self.update_projections()

#     def get_parameters(self):
#         """
#         Get parameters of the model.
#         """
#         return self.params_list

import torch
import torch.nn as nn

print(torch.cuda.is_available())

class GaLore(nn.Module):
    def __init__(self, model, rank=4, update_freq=200, galore_params=None):
        super(GaLore, self).__init__()
        self.model = model.to(torch.device("cuda"))  # Move model to GPU
        print(f"Model device: {next(self.model.parameters()).device}")  # Check model's device
        self.rank = rank
        self.update_freq = update_freq
        self.n_step = 0
        if galore_params is not None:
            self.params_list_all = galore_params
        else:
            self.params_list_all = self.model.named_parameters()
        self.params_list = [(name, param) for name, param in self.params_list_all if param.requires_grad and len(param.shape) > 1]
        self.P = {}
        self.Q = {}

        for name, param in self.params_list:
            self.P[name] = torch.empty(
                (param.data.shape[0], self.rank),
                dtype=param.data.dtype,
                device=param.data.device,
            )
            self.Q[name] = torch.empty(
                (param.data.shape[1], self.rank),
                dtype=param.data.dtype,
                device=param.data.device,
            )
            nn.init.orthogonal_(self.P[name])
            nn.init.orthogonal_(self.Q[name])

    def project(self, grad, name):
        print(f"the grad is {grad}")
        print(f"Shape of grad tensor: {grad.shape}")
        return torch.matmul(self.P[name].t(), torch.matmul(grad, self.Q[name]))

    def project_back(self, lor_grad, name):
        return torch.matmul(self.P[name], torch.matmul(lor_grad, self.Q[name].t()))

    def update_projections(self):
        with torch.no_grad():
            for name, param in self.params_list:
                grad = param.grad
                U, _, Vt = torch.linalg.svd(grad, full_matrices=False)
                self.P[name] = U[:, : self.rank]
                self.Q[name] = Vt[: self.rank, :].t()

    def step(self, update_func):
        for name, param in self.params_list:
            grad = param.grad
            # grad.retain_grad()
            lor_grad = self.project(grad, name)
            lor_update = update_func(lor_grad)
            update = self.project_back(lor_update, name)
            param.data += update
        for name, param in self.params_list_all:
            if (name, param) not in self.params_list and param.requires_grad:
                update_func(param.grad)

        self.n_step += 1
        if self.n_step % self.update_freq == 0:
            self.update_projections()

    def get_parameters(self):
        """
        Get parameters of the model.
        """
        return self.params_list
