import re

import pandas as pd


def get_num(string: str):
    return float(string.split('=')[-1])


def logAnalyse(file_path, path, after=True, after_fun=None):
    """
    日志分析
    :param file_path: 日志文件路径
    :param path: 正则表达式
    :param after 是否有后续操作
    :param after_fun 后续操作函数
    :return: 匹配结果列表
    """
    results = []
    with open(file_path, mode='r') as f:
        for i in f:
            match = re.search(path, i)
            if match:
                temp = match.group()
                if after and after_fun is not None:
                    temp = after_fun(temp)
                results.append(temp)
    return results


if __name__ == '__main__':
    # EDSR
    path1 = r'acg_loss=\d+.\d{5}'
    path2 = r'avg_psnr=\d+.\d{5}'
    path3 = r'avg_ssim=\d+.\d{5}'
    path4 = r'avg_loss=\d+.\d{5}'
    path5 = r'psnr=\d+.\d{5}'
    path6 = r'ssim=\d+.\d{5}'
    path7 = r'image \d+_\d+'
    path = r'./logs/EDSR_230428-193445.log'
    a = logAnalyse(path, path1, after_fun=get_num)
    b = logAnalyse(path, path2, after_fun=get_num)
    c = logAnalyse(path, path3, after_fun=get_num)
    d = logAnalyse(path, path4, after_fun=get_num)
    path = r'./logs/test_EDSR_230510-184628.log'
    e = logAnalyse(path, path5, after_fun=get_num)
    f = logAnalyse(path, path6, after_fun=get_num)
    g = logAnalyse(path, path7, after_fun=lambda x: x.split(' ')[-1])
    index = ['训练误差', '验证PSNR', '验证SSIM', '验证误差']
    index2 = ['测试PSNR', '测试SSIM', 'NAME']

    data1 = pd.DataFrame([a, b, c, d], index=index)
    print(data1.head())
    data1.to_excel(r'./EDSR.xlsx', sheet_name='EDSR')
    data2 = pd.DataFrame([e, f, g], index=index2)
    data2.to_excel(r'./EDSR_test.xlsx', sheet_name='EDSR_TEST')

    # RRDB
    path1 = r'avg_loss=\d+.\d{5}'
    path2 = r'psrn=\d+.\d{5}'
    path3 = r'ssim=\d+.\d{5}'
    path4 = r'val loss=\d+.\d{5}'
    path5 = r'psnr=\d+.\d{5}'
    path6 = r'ssim=\d+.\d{5}'
    path7 = r'image \d+_\d+'
    path = r'./logs/train_RRDBNet_230423-122905.log'
    a = logAnalyse(path, path1, after_fun=get_num)
    b = logAnalyse(path, path2, after_fun=get_num)
    c = logAnalyse(path, path3, after_fun=get_num)
    d = logAnalyse(path, path4, after_fun=get_num)
    path = r'./logs/test_RRDBNet_230510-173044.log'
    e = logAnalyse(path, path5, after_fun=get_num)
    f = logAnalyse(path, path6, after_fun=get_num)
    g = logAnalyse(path, path7, after_fun=lambda x: x.split(' ')[-1])
    index = ['训练误差', '验证PSNR', '验证SSIM', '验证误差']
    index2 = ['测试PSNR', '测试SSIM', 'NAME']

    data1 = pd.DataFrame([a, b, c, d], index=index)
    print(data1.head())
    data2 = pd.DataFrame([e, f, g], index=index2)
    data1.to_excel(r'./RRDB.xlsx', sheet_name='RRDB')
    data2.to_excel(r'./RRDB_test.xlsx', sheet_name='RRDB_TEST')
