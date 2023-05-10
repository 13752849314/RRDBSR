import os

import cv2

x_path = r'./data/Archive/low_res'
y_path = r'./data/Archive/high_res'
out_RRDB_path = r'./results/RRDBNet/300_300test'
out_EDSR_path = r'./results/EDSR/300_300test'
x_name = '475_4'
y_name = '475'
h, w = 300, 200

x = cv2.imread(os.path.join(x_path, x_name + '.jpg'))
x = cv2.resize(x, (h * 4, w * 4))
print(x.shape)
y = cv2.imread(os.path.join(y_path, y_name + '.jpg'))
y = cv2.resize(y, (h * 4, w * 4))
print(y.shape)
out1 = cv2.imread(os.path.join(out_RRDB_path, x_name + '.png'))
out2 = cv2.imread(os.path.join(out_EDSR_path, x_name + '.png'))

index = []
temp = 0


def mouse_callback(event, x1, y1, flags, userdata):
    global index, temp
    if event & cv2.EVENT_LBUTTONDOWN == cv2.EVENT_LBUTTONDOWN:
        index.append([x1, y1])
        print('address=', [x1, y1])
    if len(index) == 2:
        cv2.rectangle(y, index[0], index[1], (0, 0, 255), thickness=2)
        cv2.rectangle(out1, index[0], index[1], (0, 0, 255), thickness=2)
        cv2.rectangle(out2, index[0], index[1], (0, 0, 255), thickness=2)
        yf = index[0][0]
        yt = index[1][0]
        xf = index[0][1]
        xt = index[1][1]
        x_sub = x[xf:xt, yf:yt, ...]
        y_sub = y[xf:xt, yf:yt, ...]
        out1_sub = out1[xf:xt, yf:yt, ...]
        out2_sub = out2[xf:xt, yf:yt, ...]
        # 保存sub
        name = r'./images/' + f'{x_name}_{str(index)}' + '_{}.png'
        cv2.imwrite(name.format('x'), x_sub)
        cv2.imwrite(name.format('y'), y_sub)
        cv2.imwrite(name.format('RRDB'), out1_sub)
        cv2.imwrite(name.format('EDSR'), out2_sub)
        temp += 1
        index.clear()


cv2.namedWindow(f'{y_name}')
cv2.setMouseCallback(f'{y_name}', mouse_callback)
while True:
    cv2.imshow(f'{x_name}', x)
    cv2.imshow(f'{y_name}', y)
    cv2.imshow(f'{x_name}+RRDB', out1)
    cv2.imshow(f'{x_name}+EDSR', out2)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
