import os
import yaml
import numpy as np
import cv2


def generate_one_label_legend(label, color):
    img = np.ones((30, 200, 3), dtype='u1')
    img[..., :] = color
    cv2.putText(img, label, (5,img.shape[0]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0) if np.mean(color) > 128 else (255, 255, 255))
    return img


def generate_label_legends(config='cfg.yaml'):
    cfg = yaml.safe_load(open(config))
    all_label_legends = []
    for idx, label in cfg['dataset']['labels'].items():
        label_legend = generate_one_label_legend(label, cfg['dataset']['color_map'][idx])
        cv2.imwrite(os.path.splitext(config )[0] + label + ".png", label_legend)
        all_label_legends.append(label_legend)
    if not len(all_label_legends):
        return
    legend_dims = np.array((all_label_legends[0].shape[0], all_label_legends[0].shape[1]))
    desired_ratio = np.array((3, 4))
    n_ratio = desired_ratio / legend_dims
    number = len(all_label_legends)
    nimgs = np.int64(np.ceil(n_ratio * np.sqrt(number / np.prod(n_ratio))))
    bigimg = np.ones(np.int64((all_label_legends[0].shape[:2] * nimgs).tolist() + [3]), dtype='u1')
    for r in range(nimgs[0]):
        for c in range(nimgs[1]):
            idx = r*nimgs[1]+c 
            if (idx >= number):
                break
            bigimg[r*legend_dims[0]:(r+1)*legend_dims[0],
                   c*legend_dims[1]:(c+1)*legend_dims[1], :] = all_label_legends[idx]

    cv2.imwrite(os.path.splitext(config)[0] + "_legend.png", bigimg)



if __name__ == '__main__':
    import sys
    configfile = sys.argv[1]
    generate_label_legends(configfile)
