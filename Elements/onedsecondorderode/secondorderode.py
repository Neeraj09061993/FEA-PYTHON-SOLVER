import numpy as np
from gauss import gaussptwt as GS
from shapefunc import shp1dO5 as shp


class odeelement(object):
    """Calculates the element stiffness matrix and body force contributions for a bar (type) element,
        governed by the differential equation -d/dx(EAd^u/dx)+B*u=B_F. Replace E, A, B and B_F with appropriate
         quantities to solve other problems governed by same differential equation."""

    def __init__(self, NPE, NGP, ELX, ELM, ELB, ELF, time=0):
        self.NPE = NPE  # Number of Nodes per element
        self.NGP = NGP  # Number of Gauss points
        self.ELX = ELX  # The coordinates of this element
        self.ELM = ELM  # The element  properties
        self.ELB = ELB  # The B term occurring in the differential equation.
        self.ELF = ELF  # Body force term occurring in the differential equation.
        self.time = time  # time is available

    # Tangent(Stiffness for Linear Elastic material) modulus and ELM[1] is Area of cross section

    def stiff(self):
        """Returns the elemental stiffness matrix for a simple bar element with
            constant area and Young's modulus.You can replace E, A, B, F appropriately to solve heat transfer problens."""
        NPE = self.NPE  # Nodes per element
        NGP = self.NGP  # Gauss points needed for integration
        ELX = self.ELX  # Element's coordinates
        Y = self.ELM[0]  # Young's modulus
        A = self.ELM[1]  # Area of cross section
        B = self.ELB  # The term B occurring in the differential equation
        he = ELX[1] - ELX[0]  # Length of the element
        GJ = 0.500000000000 * he[0]  # Jacobian for this element
        k_glob = np.zeros((3, 3))
        Kel = np.zeros((NPE, NPE))
        G = GS.Gauss1d(NGP)
        pt, w = G.point_weight()
        for k in range(2):
            for (c, gpt) in enumerate(pt):
                Lsf = shp.Lagsf1(gpt, NPE - 1)
                wt = w[c]
                F = Lsf.f()
                DF = Lsf.fx()
                for NI in range(NPE):
                    for NJ in range(NPE):
                        Kel[NI][NJ] = Kel[NI][NJ] + Y * A * DF[NI] / GJ * DF[NJ] / GJ * wt * GJ + B * F[NI] * F[NJ] * wt * GJ
        k_glob[NI+ k][NJ + k] = k_glob[NI + k][NJ + k] + Kel[NI][NJ]
        return k_glob

    def bodyforce(self):
        """Returns the body force vector for a simple bar element"""
        NPE = self.NPE  # Nodes per element
        NGP = self.NGP  # Gauss points needed for integration
        ELX = self.ELX  # Element's coordinates
        B_F = self.ELF  # The Body force occurring in the differential equation
        he = ELX[1] - ELX[0]  # Length of the element
        GJ = 0.500000000000 * he[0]  # Jacobian for this element
        Fbel = np.zeros((NPE))
        G = GS.Gauss1d(NGP)
        pt, w = G.point_weight()
        for i in range(NGP):
            Lsf = shp.Lagsf1(pt[i], NPE - 1)
            wt = w[i]
            F = Lsf.f()
            DF = Lsf.fx()
            # Xsq = 0
            # for p in range(NPE):
            # Xsq = Xsq + F[p] * ELX[p] * ELX[p]
            for NI in range(NPE):
                Fbel[NI] = Fbel[NI] + B_F * F[NI] * wt * GJ
                # Fbel[NI] = Fbel[NI] - Xsq * F[NI] * wt * GJ
        return Fbel

    # def massmatrix(self):
    #     NPE = self.NPE  # Nodes per element
    #     NGP = self.NGP  # Gauss points needed for integration
    #     ELX = self.ELX  # Element's coordinates
    #     cm = self.ELM[2]  # Coefficient of do(u)/do(t)
    #     he = ELX[1] - ELX[0]  # Length of the element
    #     GJ = 0.500000000000 * he  # Jacobian for this element
    #     Mel = np.zeros((NPE, NPE))
    #     G = GS.Gauss1d(NGP)
    #     pt, w = G.point_weight()
    #     for (c, gpt) in enumerate(pt):
    #         Lsf = shp.Lagsf1(gpt, NPE - 1)
    #         wt = w[c]
    #         F = Lsf.f()
    #         for NI in range(NPE):
    #             for NJ in range(NPE):
    #                 Mel[NI][NJ] = Mel[NI][NJ] + wt * cm * F[NI] * F[NJ] * GJ
    #     return Mel


# if __name__ == "__main__":
#     K = np.array([0, 1.0])
#     P = np.array([1.0, 1.0, 1.0])
#     bar = odeelement(2, 3, K, P, 1.0, 3.0)
#     print(bar.stiff())
#     print(bar.bodyforce())
#     print(bar.massmatrix())
