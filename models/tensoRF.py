from .tensorBase import *
from .sample_utils import *

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.coarse_density_volume = self.init_volume(self.sigma_dim, self.gridSize, device)

        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_volume(self, channel, gridSize, device):
        feature_volume = [torch.nn.Parameter(torch.randn((1, channel, gridSize[2], gridSize[1], gridSize[0])))]
        return torch.nn.ParameterList(feature_volume).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.coarse_density_volume, 'lr': lr_init_spatialxyz},
                     
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, 
                     {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        
        if isinstance(self.densityModule, torch.nn.Module):
            grad_vars += [{'params':self.densityModule.parameters(), 'lr':lr_init_network}]
        
        return grad_vars

    
    def density_L1(self):
        total = 0
        for idx in range(len(self.coarse_density_volume)):
            total = total + torch.mean(torch.abs(self.coarse_density_volume[idx]))
        
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.coarse_density_volume)):
            total = total + reg(self.coarse_density_volume[idx]) * 1e-2
        
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2
        return total

    def compute_densityfeature(self, xyz_sampled):
        coordinate = xyz_sampled[...].view(1, -1, 1, 1, 3)
        sigma_feature_coarse = grid_sample_3d(self.coarse_density_volume[0], coordinate,
                                            ).view(-1, *xyz_sampled.shape[:1])
        
        return sigma_feature_coarse.transpose(0, 1) # [N, feat_dim]


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)



    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.coarse_density_volume[0] = torch.nn.Parameter(F.interpolate(self.coarse_density_volume[0].data, size=(res_target[2], res_target[1], res_target[0]), mode='trilinear',
                              align_corners=True))

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    # @torch.no_grad()
    # def shrink(self, new_aabb):
    #     print("====> shrinking ...")
    #     xyz_min, xyz_max = new_aabb
    #     t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
    #     t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
    #     b_r = torch.stack([b_r, self.gridSize]).amin(0)

    #     self.density_volume[0] = torch.nn.Parameter(
    #             self.density_volume[0].data[...,t_l[2]:b_r[2],t_l[1]:b_r[1],t_l[0]:b_r[0]]
    #         )
        
    #     for i in range(len(self.vecMode)):
    #         mode0 = self.vecMode[i]
    #         self.app_line[i] = torch.nn.Parameter(
    #             self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
    #         )
    #         mode0, mode1 = self.matMode[i]
    #         self.app_plane[i] = torch.nn.Parameter(
    #             self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
    #         )

    #     newSize = b_r - t_l
    #     self.aabb = new_aabb
    #     self.update_stepSize((newSize[0], newSize[1], newSize[2]))
