
import numpy as np
from AARC import inventory_aarc

if __name__ == '__main__':
    c = [0.1, 0.1, 0.1, 0.1, 0.1]
    h = [0.02, 0.02, 0.02, 0.02, 0.02]
    b = [0.2, 0.2, 0.2, 0.2, 2]
    d_0 = [200, 200, 200, 200, 200]
    r = 100
    T = 5

    pb, solution = inventory_aarc(c, h, b, d_0, r)
    print("Optimal value: {}".format(pb.obj_value()))
