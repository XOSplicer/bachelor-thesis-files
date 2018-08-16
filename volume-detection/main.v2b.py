import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import cv2

def R_x(a):
    return np.matrix([ \
        [1., 0., 0.], \
        [0., math.cos(a), -math.sin(a)], \
        [0., math.sin(a), math.cos(a)] \
    ])

def R_y(a):
    return np.matrix([ \
        [math.cos(a), 0., math.sin(a)], \
        [0., 1., 0.], \
        [-math.sin(a), 0., math.cos(a)] \
    ])

def R_z(a):
    return np.matrix([ \
        [math.cos(a), -math.sin(a), 0.], \
        [math.sin(a), math.cos(a), 0.], \
        [0., 0., 1.] \
    ])

OUTPUT = False

CAPTURES = [
    {
        'name': 'CAM_A',
        'file': 'input/05_synced/01_CAM_A.mp4',
        'camera_matrix': [
            [954.7283518831631, 0.0, 328.38251204771797],
            [0.0, 954.3239686284525, 682.7462363926942],
            [0.0, 0.0, 1.0]
        ],
        'dist_coefs': [0.18274165220464775, -0.5011242764992531, 0.0005626441108219812, -0.003632685308848848, 0.4201469506894296],
        'coord_m': [0., 0.8, 1.5],
        'view_rot_n': [1.0, 0., 0.],
        'view_rot_matrix': R_z(math.pi) @ R_y(math.pi / 2),
        '3d_p_size': [1.28, 0.72],
    },
    {
        'name': 'CAM_B',
        'file': 'input/05_synced/01_CAM_B.mp4',
        'camera_matrix': [
            [1044.1788361145225, 0.0, 373.9586078294486],
            [0.0, 1045.4307249439385, 656.1732490760546],
            [0.0, 0.0, 1.0]
        ],
        'dist_coefs': [0.08568675847412435, 0.06816541721676729, 0.004349902176286388, -0.004190652643689862, 0.40310740671385564],
        'coord_m': [0.65, 1.65, 1.5],
        'view_rot_n': [0., -1., 0.],
        'view_rot_matrix': R_z(math.pi / 2) @ R_y(math.pi / 2),
        '3d_p_size': [1.28, 0.72],
    },
    {
        'name': 'CAM_C',
        'file': 'input/05_synced/01_CAM_C.mp4',
        'camera_matrix': [
            [1030.803754551308, 0.0, 377.4660646383369],
            [0.0, 1030.625958017089, 642.3814181010922],
            [0.0, 0.0, 1.0]
        ],
        'dist_coefs': [0.0493844563634977, -0.01846929734900904, 0.0039138050833449605, 0.011433153710392586, 0.044727206326660544],
        'coord_m': [1.3, 0.8, 1.5],
        'view_rot_n': [-1.0, 0., 0.],
        'view_rot_matrix': R_y(math.pi / 2),
        '3d_p_size': [1.28, 0.72],
    },
    {
        'name': 'CAM_D',
        'file': 'input/05_synced/01_CAM_D.mp4',
        'camera_matrix': [
            [1323.8337374168664, 0.0, 695.6961612404053],
            [0.0, 1338.54917998936, 552.8508654004544],
            [0.0, 0.0, 1.0]
        ],
        'dist_coefs': [-0.29913419744762193, 0.4489268421257414, -0.0015609838206597788, -0.004756895033024813, -0.5753214907483867],
        'coord_m': [0.65, 0., 2.2],
        'view_rot_n': [0., 1., -1.4 ], # theta=54.5deg
        'view_rot_matrix':   R_x(math.atan(1.4)) @ R_z(-math.pi / 2),
        '3d_p_size': [0.72, 1.28],
    }
]

ROOM_SIZE_M = {
    'width': 1.30,
    'depth': 1.65,
    'height': 2.30
}

ROOM_DIM_VX = {
    'width': int(40 / 2),
    'depth': int(50 / 2),
    'height': int(60 /2)
}

ROOM_SCALE_MPVX = {
    'width': ROOM_SIZE_M['width'] / ROOM_DIM_VX['width'],
    'depth': ROOM_SIZE_M['depth'] / ROOM_DIM_VX['depth'],
    'height': ROOM_SIZE_M['height'] / ROOM_DIM_VX['height']
}

def nop(x):
    pass

# also https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d#476311
def project_2d_point_to_3d_cam_plane(point, cam, localOffset=np.array([0., 0., 0.])):
    point =  np.array([point[0], point[1], 0.]).T + localOffset # translate localy, set in 3d
    projected = cam['view_rot_matrix'] @ point
    projected = projected + cam['coord_m']
    return projected

def multi_preview(mats):
    height = 240
    scales = [ height / m.shape[0] for m in mats]
    scaled = [ cv2.resize(mats[i], (0,0),fx=scales[i], fy=scales[i]) for i in range(len(mats))]
    dst = np.hstack(tuple(scaled))
    return dst

def init_3d_plot():
    pg.mkQApp()
    view = gl.GLViewWidget()
    view.show()
    xgrid = gl.GLGridItem()
    ygrid = gl.GLGridItem()
    zgrid = gl.GLGridItem()
    view.addItem(xgrid)
    view.addItem(ygrid)
    view.addItem(zgrid)
    xgrid.rotate(90, 0, 1, 0)
    ygrid.rotate(90, 1, 0, 0)
    xgrid.scale(0.2, 0.1, 0.1)
    ygrid.scale(0.2, 0.1, 0.1)
    zgrid.scale(0.1, 0.2, 0.1)
    return view

def create_3d_room(view):
    x = ROOM_SIZE_M['width']
    y = ROOM_SIZE_M['depth']
    z = ROOM_SIZE_M['height']
    o = 0.
    room_lines = [
        [o, o, o],[x, o, o],
        [o, o, o],[o, y, o],
        [o, o, o],[o, o, z],
        [x, y, z],[o, y, z],
        [x, y, z],[x, o, z],
        [x, y, z],[x, y, o]
     ]
    room_plot = gl.GLLinePlotItem(pos=np.array(room_lines), \
        color=(1.0, 1.0, 1.0, 1.0), width=2, antialias=False, mode='lines')
    view.addItem(room_plot)

def create_3d_cams(view):
    cam_coords = np.array([ c['coord_m'] for c in CAPTURES ])
    cam_plot = gl.GLScatterPlotItem(pos=cam_coords, color=(1.0, 0.0, 0.0, 1.0), size=20, pxMode=True)
    view.addItem(cam_plot)

def create_3d_volume(view):
    volume_plot = gl.GLScatterPlotItem(pos=np.array([]), color=(0.0, 1.0, 0.0, 1.0), size=10, pxMode=True)
    view.addItem(volume_plot)
    return volume_plot

def create_3d_side(view):
    side_plot = gl.GLScatterPlotItem(pos=np.array([]), color=(0.0, 0.5, 1.0, 1.0), size=5, pxMode=True)
    view.addItem(side_plot)
    return side_plot

def print_versions():
    print('python=', sys.version)
    print('opencv=', cv2.__version__)
    print('numpy=', np.__version__)
    print('matplotlib=', matplotlib.__version__)
    print('pyqtgraph=', pg.__version__)

# https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
def convertQImageToMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
    return arr

def main():
    print_versions()

    print(CAPTURES)

    cv2.namedWindow('sliders')
    cv2.createTrackbar('gaussian_kernel','sliders',11,51,nop)
    cv2.createTrackbar('open_kernel','sliders',3,51,nop)
    cv2.createTrackbar('dilate_kernel','sliders',17,51,nop)

    view_3d = init_3d_plot()
    create_3d_room(view_3d)
    create_3d_cams(view_3d)
    vol_3d = create_3d_volume(view_3d)
    side_3d = create_3d_side(view_3d)

    caps = [ cv2.VideoCapture(c['file']) for c in CAPTURES ]
    # bgss = [ cv2.createBackgroundSubtractorMOG2() for _ in caps ]
    bgss_pp = [ cv2.createBackgroundSubtractorMOG2() for _ in caps ]

    video_out_res = (1920, 480)
    if OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter('output.avi',fourcc, 25.0, video_out_res)

    while True:
        # read all capture images at this moment
        raws = [ c.read()[1] for c in caps]

        # convert to grayscale
        gry = [ cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in raws ]
        gry_preview = multi_preview(gry)
        #cv2.imshow('gry_preview', gry_preview)

        # get parameters from trackbars
        g_size = cv2.getTrackbarPos('gaussian_kernel','sliders')
        g_size = g_size - g_size % 2 +1
        o_size = cv2.getTrackbarPos('open_kernel','sliders')
        o_size = o_size - o_size % 2 +1
        d_size = cv2.getTrackbarPos('dilate_kernel','sliders')
        d_size = d_size - d_size % 2 +1

        # pre process images and appy background subtractor
        pp = [ cv2.GaussianBlur(r, (g_size,g_size), 0) for r in raws ]
        bgs_pp_masks = [ bgss_pp[i].apply(pp[i]) for i in range(len(raws)) ]

        # post process foreground masks
        open_kernel = np.ones((o_size,o_size),np.uint8)
        dilate_kernel = np.ones((d_size,d_size),np.uint8)
        bgs_pp_masks_t = [ cv2.threshold(b, 254, 255, cv2.THRESH_BINARY)[1] for b in bgs_pp_masks]
        bgs_pp_masks_b = [ cv2.dilate(cv2.morphologyEx(b, cv2.MORPH_OPEN, open_kernel), dilate_kernel, iterations = 10)\
             for b in bgs_pp_masks_t ]
        bgs_pp_preview = multi_preview(bgs_pp_masks_b)
        #cv2.imshow('bgs_pp_preview', bgs_pp_preview)

        # bounding box test for 3d preview
        bounding_points = np.array([ project_2d_point_to_3d_cam_plane(\
                np.array(p), \
                c, \
                localOffset=np.array([-c['3d_p_size'][0] / 2, -c['3d_p_size'][1] / 2, 0.0]) \
            ) \
            for c in CAPTURES \
            for p in [[0., 0.], [c['3d_p_size'][0], 0.], [0., c['3d_p_size'][1]], [c['3d_p_size'][0], c['3d_p_size'][1]]] ])
        resized_masks = [ \
            cv2.resize(mask, (0,0), fx = 1/40, fy = 1/40, interpolation=cv2.INTER_AREA) \
            for mask in bgs_pp_masks_b ]
        preview_points = np.array([ \
            project_2d_point_to_3d_cam_plane(\
                np.array([u / resized_masks[i].shape[0] * CAPTURES[i]['3d_p_size'][0], v / resized_masks[i].shape[1] * CAPTURES[i]['3d_p_size'][1]]), \
                CAPTURES[i], \
                localOffset=np.array([-CAPTURES[i]['3d_p_size'][0] / 2, -CAPTURES[i]['3d_p_size'][1] / 2, 0.0]) \
            ) \
            for i in range(len(CAPTURES)) \
            for u in range(resized_masks[i].shape[0]) \
            for v in range(resized_masks[i].shape[1]) \
            if resized_masks[i][u][v] == 255 \
            ])

        side_points = np.concatenate((bounding_points, preview_points)) if preview_points.size else bounding_points
        side_3d.setData(pos=side_points)




        # create combined preview
        image_3d = convertQImageToMat(view_3d.readQImage())
        scale = 480 / image_3d.shape[0]
        image_3d_preview = cv2.resize(image_3d, (0,0), fx=scale, fy=scale)
        image_3d_preview = cv2.cvtColor(image_3d_preview, cv2.COLOR_BGRA2BGR)
        # cv2.imshow('image_3d_preview', image_3d_preview)
        full_preview = np.hstack((\
            cv2.cvtColor(np.vstack((gry_preview, bgs_pp_preview)), cv2.COLOR_GRAY2BGR), \
            image_3d_preview \
            ))
        cv2.imshow('full_preview', full_preview)
        out = np.zeros((video_out_res[1], video_out_res[0], 3), dtype=np.uint8)
        out[:, 0:full_preview.shape[1], :] = full_preview
        #cv2.imshow('out', out)
        if OUTPUT:
            video_out.write(out)


        # unnecessary distance measure
        # canny_dists = [cv2.distanceTransform(cv2.bitwise_not(cv2.Canny(m, 0, 255)), cv2.DIST_L2, 3) for m in bgs_pp_masks_b]
        # canny_dists_preview = multi_preview(canny_dists)
        # cv2.imshow('canny_dists_preview', canny_dists_preview)
        # print(cannys[0].shape, cannys[0].dtype)

        if cv2.waitKey(int(1000/25)) != -1:
            break

    cv2.destroyAllWindows()
    for c in caps:
        c.release()
    if OUTPUT:
        video_out.release()
    plt.close()

if __name__ == '__main__':
    main()