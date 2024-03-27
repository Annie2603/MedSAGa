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
            self.params_list = galore_params
            print(f"Initial length of params_list: {len(list(self.params_list))}")
        else:
            self.params_list = self.model.named_parameters()
            
        print(len(list(self.model.named_parameters())))
        # self.params_list = [(name, param) for name, param in params_list_temp if param.requires_grad and len(param.shape) > 1]
        self.P = {}
        self.Q = {}
        count = 0
        for name, param in self.params_list:
            if param.requires_grad and len(param.shape) > 1:
                count +=1
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
        print(f"initially the count is {count}")

    def project(self, grad, name):
        # print(f"the grad is {grad}")
        # print(f"Shape of grad tensor: {grad.shape}")
        return torch.matmul(self.P[name].t(), torch.matmul(grad, self.Q[name]))

    def project_back(self, lor_grad, name):
        return torch.matmul(self.P[name], torch.matmul(lor_grad, self.Q[name].t()))

    def update_projections(self):
        with torch.no_grad():
            for name, param in self.params_list:
                if param.requires_grad:
                    if len(param.shape) > 1:
                        # if name == 'image_encoder.pos_embed':
                        #     print(f"it is present {param.shape}, {self.Q[name].shape}")
                        grad = param.grad
                        U, _, Vt = torch.linalg.svd(grad, full_matrices=False)
                        self.P[name] = U[:, : self.rank]
                        self.Q[name] = Vt[: self.rank, :].t()

    def step(self, update_func):
        count1 = 0
        print(f"the length of the params list is {len(list(self.params_list))}")
        for name, param in self.params_list:
            # print(f"printing the gradient {param.grad}")
            grad = param.grad
            if param.requires_grad:
                if len(param.shape) > 1:
                    count1 += 1
                    # grad.retain_grad()
                    lor_grad = self.project(grad, name)
                    lor_update = update_func(lor_grad)
                    update = self.project_back(lor_update, name)
                    param.data += update
                else:
                    update_func(param.grad)
        print(f"later the count is {count1}")
        self.n_step += 1
        if self.n_step % self.update_freq == 0:
            self.update_projections()

    def get_parameters(self):
        """
        Get parameters of the model.
        """
        return self.params_list

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
#             self.params_list_all = galore_params
#         else:
#             self.params_list_all = self.model.named_parameters()
#         self.params_list = [(name, param) for (name, param) in self.params_list_all if param.requires_grad and len(param.shape) == 2]
#         self.P = {}
#         self.Q = {}

#         for name, param in self.params_list:
#             self.P[name] = torch.empty(
#                 (param.data.shape[0], self.rank),
#                 dtype=param.data.dtype,
#                 device=param.data.device,
#             )
#             self.Q[name] = torch.empty(
#                 (param.data.shape[1], self.rank),
#                 dtype=param.data.dtype,
#                 device=param.data.device,
#             )
#             nn.init.orthogonal_(self.P[name])
#             nn.init.orthogonal_(self.Q[name])
#         name1 = self.params_list[0][0]
#         print(f"checking: {name1}, {self.params_list[0][1].shape}, {self.Q[name1].shape}")
#         # for name, param in self.params_list:
#         #     print(f"{name}: {param.shape}, {self.Q[name].shape}")
#         for key in self.Q.keys():
#             if self.Q[key].shape[0] == 32:
#                 print(f"found: the name is {key}") 
#         print("done checking")
#         for name, param in self.params_list:
#             if name == 'mask_decoder.output_upscaling.3.weight':
#                 print(f"mask_decoder.output_upscaling.3.weight {param.shape}")

#     def project(self, grad, name):
#         # print(f"the grad is {grad}")
#         # print(f"Shape of grad tensor: {grad.shape}")
#         return torch.matmul(self.P[name].t(), torch.matmul(grad, self.Q[name]))

#     def project_back(self, lor_grad, name):
#         return torch.matmul(self.P[name], torch.matmul(lor_grad, self.Q[name].t()))

#     def update_projections(self):
#         with torch.no_grad():
#             for name, param in self.params_list:
#                 grad = param.grad
#                 U, _, Vt = torch.linalg.svd(grad, full_matrices=False)
#                 self.P[name] = U[:, : self.rank]
#                 self.Q[name] = Vt[: self.rank, :].t()

#     def step(self, update_func):
#         for name, param in self.params_list:
#             print('inside the first')
#             grad = param.grad
#             # grad.retain_grad()
#             lor_grad = self.project(grad, name)
#             lor_update = update_func(lor_grad)
#             update = self.project_back(lor_update, name)
#             param.data += update
#         print('outside the first')
#         for name, param in self.params_list_all:
#             if (name, param) not in self.params_list and param.requires_grad:
#                 update_func(param.grad)

#         self.n_step += 1
#         if self.n_step % self.update_freq == 0:
#             self.update_projections()

#     def get_parameters(self):
#         """
#         Get parameters of the model.
#         """
#         return self.params_list