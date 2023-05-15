import numpy as np

class readmesh(object):

    def __init__(self, filename):
        self.filename = filename
        self.mesh = {}
        self.physical = {}
        self.info = {}

    def readmshfile(self):
        fname = self.filename
        file = open(fname, 'r')

        for j in range(3):
            k = file.readline()

        k = file.readline()

        if k == '$PhysicalNames\n':
            num = int(file.readline())
            physical_tags = {}
            p = 0
            l = 0
            s = 0
            for j in range(num):
                k = file.readline().split()
                if int(k[0]) == 0:
                    name = "point_name"
                    p = p + 1
                if int(k[0]) == 1:
                    name = "line_name"
                    l = l + 1
                if int(k[0]) == 2:
                    name = "surface_name"
                    s = s + 1
                physical_tags.update({int(k[1]): k[2].replace('"', '')})

        k = file.readline()
        k = file.readline()

        if k == "$Nodes\n":
            numnodes = int(file.readline())
            nid = np.zeros(numnodes, dtype=int)
            x = np.zeros(numnodes, dtype=float)
            y = np.zeros(numnodes, dtype=float)
            z = np.zeros(numnodes, dtype=float)




            for j in range(numnodes):
                k = file.readline().split()
                nid[j] = int(k[0])
                x[j] = float(k[1])
                y[j] = float(k[2])
                z[j] = float(k[3])

        x = x.reshape((numnodes, 1))
        y = y.reshape((numnodes, 1))
        z = z.reshape((numnodes, 1))

        xnorm = np.linalg.norm(x)
        ynorm = np.linalg.norm(y)

        if ynorm == 0:
            dim = 1
        if ynorm != 0:
            dim = 2

        print("Dimension is :", dim)

        coords = np.concatenate((x, y, z), axis=1)

        k = file.readline()
        k = file.readline()

        if k == "$Elements\n":
            numelements = int(file.readline())
            el_id = np.zeros(numelements, dtype=int)
            el_type = np.zeros(numelements, dtype=int)
            el_ph_tags = np.zeros(numelements, dtype=int)
            el_line = 0
            el_quad = 0
            el_tri = 0
            el_point = 0
            global_connec = []

            for j in range(numelements):
                k = file.readline().split()
          
                if int(k[1]) == 15:  # a point element
                    npe = 1
                    el_type[j] = 15
                    el_point = el_point + 1

                if int(k[1]) == 1:  # linear line
                    npe = 2
                    el_type[j] = 0
                    el_line = el_line + 1

                if int(k[1]) == 8:  # quadratic line
                    npe = 3
                    el_type[j] = 0
                    el_line = el_line + 1
                if int(k[1]) == 26:  #  cubic line
                    npe = 4
                    el_type[j] = 0
                    el_line = el_line + 1

                if int(k[1]) == 2:  # Linear Triangle
                    npe = 3
                    el_type[j] = 2
                    el_tri = el_tri + 1

                if int(k[1]) == 3:  # Linear Quadrilateral
                    npe = 4
                    el_type[j] = 1
                    el_quad = el_quad + 1
                    26


                if int(k[1]) == 9:  # Quadratic Triangle
                    npe = 6
                    el_type[j] = 2
                    el_tri = el_tri + 1

                if int(k[1]) == 10:  # Quadratic Quadrilateral
                    npe = 9
                    el_type[j] = 1
                    el_quad = el_quad + 1

                if int(k[1]) == 21:  # Cubic Triangle
                    npe = 10
                    el_type[j] = 2
                    el_tri = el_tri + 1

                el_id[j] = int(k[0])
                el_ph_tags[j] = int(k[3])
                int_nodes = [int(l) - 1 for l in k[int(k[2]) + 3:]]
                global_connec.append(np.array(int_nodes))

        uniq_el = np.unique(el_type)
        dict_nodes = {}

        subtract = 1 + el_point + (dim - 1) * el_line

        for h in uniq_el:
            K = np.where(el_type == h)
            for (item, key) in enumerate(physical_tags):
                J = np.where(el_ph_tags == key)
                T = np.intersect1d(K, J)
                gc = [global_connec[i] for i in T]
                if dim == 1:
                    if T.size != 0 and h == 15:
                        uniqgc = np.unique(np.asarray(np.ravel(gc)))
                        dict_nodes.update({physical_tags[key]: uniqgc})
                    if T.size != 0 and h != 15:
                        uniqgc = [i - subtract for i in el_id[T]]
                        dict_nodes.update({physical_tags[key]: uniqgc})
                if dim == 2:
                    if T.size != 0 and (h == 15 or h == 0):
                        uniqgc = np.unique(np.asarray(np.ravel(gc)))
                        dict_nodes.update({physical_tags[key]: uniqgc})
                    if T.size != 0 and (h != 15 and h != 0):
                        uniqgc = [i - subtract for i in el_id[T]]
                        dict_nodes.update({physical_tags[key]: uniqgc})

        triangles = []
        quads = []
        lines = []
        points = []
        for n in range(numelements):
            tpe = el_type[n]
            if tpe == 15:
                points.append(global_connec[n])
            if tpe == 0:
                lines.append(global_connec[n])
            if tpe == 1:
                quads.append(global_connec[n])
            if tpe == 2:
                triangles.append(global_connec[n])

        dict_connec = {"Points": np.asarray(points), "Lines": np.asarray(lines), "Quads": np.asarray(quads),
                       "Triangles": np.asarray(triangles)}
        print("Number of 2D elements:", el_tri + el_quad)
        print("Number of 1D elements:", el_line)
        print("Number of 0D elements:", el_point)

        if dim == 1:
            global_connectivity = global_connec[el_point:]
        if dim == 2:
            global_connectivity = global_connec[el_point + el_line:]

        nodes_per_triangle = 3
        nodes_per_quad = 4

        self.physical = dict_nodes
        self.mesh = {"Type": el_type, "Connectivity": dict_connec, "Coords": coords, "Physical_IDS": self.physical,
                     "Full_Connectivity_dim": global_connectivity}
        self.info = {"Num_nodes": numnodes,"Nodes_per_elem": npe, "Num_elem": el_tri + el_quad + el_line, "Triangles": el_tri,
                     "Quads": el_quad, "Lines": el_line, "npt": nodes_per_triangle, "npq": nodes_per_quad}


if __name__ == "__main__":
    filename = "Line_mesh.msh"
    K = readmesh(filename=filename)
    K.readmshfile()

