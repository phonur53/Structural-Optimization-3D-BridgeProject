import matplotlib.pyplot as plt
import torch
from matplotlib import cm, colors


class Truss:
    def __init__(self, nodes, elements, forces, displacements, constraints, areas, moduli):
        self.nodes = nodes
        self.elements = elements
        self.forces = forces
        self.displacements = displacements
        self.constraints = constraints
        self.areas = areas
        self.moduli = moduli

    def bar_vectors(self):
        return self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]

    def lengths(self):
        return torch.linalg.norm(self.bar_vectors(), dim=-1)

    def volume(self):
        return torch.inner(self.areas, self.lengths())

    def stiffness(self):
        dx = self.bar_vectors()
        l = torch.linalg.norm(dx, dim=-1)
        n = dx / l[:, None]

        k = (self.moduli * self.areas / l)[:, None, None] * torch.einsum("ei,ej->eij", n, n)
        ke = torch.cat([torch.cat([k, -k], dim=2), torch.cat([-k, k], dim=2)], dim=1)

        dofs = (3 * self.elements[:, :, None] + torch.arange(3)).reshape(-1, 6)
        rows = dofs[:, :, None].expand(-1, 6, 6).reshape(-1)
        cols = dofs[:, None, :].expand(-1, 6, 6).reshape(-1)

        size = self.nodes.numel()
        K = torch.zeros(size, size, dtype=self.nodes.dtype)
        return K.index_put((rows, cols), ke.reshape(-1), accumulate=True)

    def solve(self):
        K = self.stiffness()
        free = ~self.constraints.ravel()
        u = self.displacements.ravel().clone()
        f_ext = self.forces.ravel()

        rhs = f_ext[free] - K[free][:, ~free] @ u[~free]
        u[free] = torch.linalg.solve(K[free][:, free], rhs)
        f = K @ u
        return u.reshape(-1, 3), f.reshape(-1, 3)

    def compute_stress(self, u):
        dx = self.bar_vectors()
        du = u[self.elements[:, 1]] - u[self.elements[:, 0]]
        strain = torch.sum(du * dx, dim=-1) / torch.sum(dx * dx, dim=-1)
        return self.moduli * strain

    @torch.no_grad()
    def plot(self, u=None, sigma=None, scale=1.0, force_size_factor=0.1, title=None, sigma_limit=None):
        fig = plt.figure(figsize=(12, 4.5))
        # 3D axes are always square, so make them taller than the figure to fill the width
        ax = fig.add_axes([0.0, -0.9, 0.88, 2.8], projection="3d")

        pos = self.nodes.detach().clone()
        if u is not None:
            pos += scale * u.detach()

        if sigma is not None:
            lim = sigma_limit or float(sigma.abs().max())
            norm = colors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
            cmap = plt.get_cmap("coolwarm")
            bar_colors = cmap(norm(sigma.detach().numpy()))
        else:
            bar_colors = ["#4a4a4a"] * len(self.elements)

        for (a, b), c in zip(self.elements.tolist(), bar_colors):
            ax.plot(*pos[[a, b]].T.numpy(), color=c, linewidth=2)

        supports = self.constraints.all(dim=1)
        ax.scatter(*pos[supports].T.numpy(), color="black", marker="^", s=80,
                   label="Support", depthshade=False)

        loaded = self.forces.abs().sum(dim=1) > 0
        p = pos[loaded].numpy()
        f = (force_size_factor * self.forces[loaded] / self.forces.abs().max()).numpy()
        ax.quiver(*(p - f).T, *f.T, color="#2e7d32", arrow_length_ratio=0.35,
                  linewidth=1.2, label="Load")

        if sigma is not None:
            cax = fig.add_axes([0.88, 0.2, 0.012, 0.6])
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                         label="Axial stress (compression < 0 < tension)")

        lo = pos.min(dim=0).values.numpy() - 0.1
        hi = pos.max(dim=0).values.numpy() + 0.1
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=18, azim=-62)
        ax.set_axis_off()

        if title:
            fig.suptitle(title, y=0.96)
        fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95), frameon=False)
        return ax
