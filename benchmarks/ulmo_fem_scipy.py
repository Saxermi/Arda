"""SciPy baseline for the 64 x 64 Darcy problem in examples/ulmo_fem.arda."""
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg


def main():
    width = 64
    start = perf_counter()
    grid = np.linspace(0, 1, width)
    x, y = np.meshgrid(grid, grid)
    nodes = np.column_stack((x.ravel(), y.ravel()))
    idx = np.arange(width**2).reshape(width, width)[:-1, :-1].ravel()
    cells = np.stack((
        np.column_stack((idx, idx+1, idx+width+1)),
        np.column_stack((idx, idx+width+1, idx+width)),
    ), axis=1).reshape(-1, 3)
    fixed = (nodes[:, 0] == 0) | (nodes[:, 0] == 1)
    created = perf_counter()
    vertices = nodes[cells]
    x, y = (vertices - np.roll(vertices, 1, axis=1)).transpose(2, 0, 1)
    # Right triangles of equal area. The edge opposite vertex i is edge i+2.
    area = 0.5/(width-1)**2
    x, y = np.roll(x, 1, axis=1), np.roll(y, 1, axis=1)
    local = (x[:, :, None]*x[:, None, :] + y[:, :, None]*y[:, None, :])/(4*area)
    rows = np.broadcast_to(cells[:, :, None], local.shape)
    cols = np.broadcast_to(cells[:, None, :], local.shape)
    free = ~fixed[rows]
    keep = free & ~fixed[cols] & (local != 0)
    bd = np.flatnonzero(fixed)
    stiffness = coo_matrix(
        (np.r_[local[keep], np.ones(bd.size)], (np.r_[rows[keep], bd], np.r_[cols[keep], bd])),
        shape=(len(nodes), len(nodes)),
    ).tocsr()
    prescribed = np.where(fixed, nodes[:, 0], 0)
    rhs = prescribed.copy()
    mask = free & fixed[cols]
    np.add.at(rhs, rows[mask], -local[mask]*prescribed[cols[mask]])
    assembled = perf_counter()
    pc = diags(1.0/stiffness.diagonal()).tocsr()
    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    pressure, status = cg(stiffness, rhs, M=pc, rtol=1e-10, maxiter=1000, callback=count)
    elapsed = perf_counter()
    print("Vertices, triangles, CSR entries", len(nodes), len(cells), stiffness.nnz)
    print("Converged, iterations, relative residual", status == 0, iterations,
          np.linalg.norm(stiffness @ pressure - rhs)/np.linalg.norm(rhs))
    print("Relative nodal error", np.linalg.norm(pressure-nodes[:, 0])/np.linalg.norm(nodes[:, 0]))
    print("Mesh, assembly, CG milliseconds", 1000*(created-start), 1000*(assembled-created),
          1000*(elapsed-assembled))


if __name__ == "__main__":
    main()
