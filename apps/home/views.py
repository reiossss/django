# -*- encoding: utf-8 -*-

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from concurrent.futures import ThreadPoolExecutor
import joblib
import pandas as pd
import re
import os
import nltk
from nltk.tokenize import word_tokenize

# 下载nltk所需的词汇库
nltk.download('punkt')

import math


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1])
    sum2 = sum([vec2[x]**2 for x in vec2])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator


def edit_distance(str1, str2):
    """
    计算两个字符串的编辑距离
    """
    m = len(str1)
    n = len(str2)

    # 创建一个二维数组来存储子问题的解
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充 dp 表格
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 没有修改
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                               dp[i][j - 1] + 1,    # 插入
                               dp[i - 1][j - 1] + 1) # 替换

    return dp[m][n]


def remove_comments(code):
    """
    去除源代码中的注释部分（包括单行注释和多行注释）
    """
    code = re.sub(r'//.*', '', code)  # 去除单行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # 去除多行注释
    return code


def remove_extra_spaces(code):
    """
    移除代码中的多余空格和空行
    """
    code = re.sub(r'\s+', ' ', code)  # 将多个空格替换为一个空格
    code = re.sub(r'^\s+|\s+?$', '', code)  # 去除行首和行尾空格
    return code


def tokenize_code(code):
    """
    对代码进行分词，返回一个词汇列表
    """
    # 使用nltk的word_tokenize进行分词
    return word_tokenize(code)


def standardize_code(code):
    """
    完整的代码标准化流程：去注释、去多余空格、分词
    """
    # with open(file_path, 'r') as file:
    #     code = file.read()
    #
    # 去注释
    code = remove_comments(code)

    # 去多余空格
    code = remove_extra_spaces(code)

    # 分词
    tokens = tokenize_code(code)

    return tokens


def jaccard_similarity(set1, set2):
    """
    计算两个集合的 Jaccard 相似度
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


executor = ThreadPoolExecutor(2)


@login_required(login_url="/login/")
def index(request):
    context = {}
    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    try:
        load_template = request.path.split('/')[-1]
        print("load_template:", load_template)
        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template
        if load_template == 'deep_model_code_contrast.html':
            code1 = request.GET.get('code1', '')
            code2 = request.GET.get('code2', '')
            if code1 and code2:
                tokens_1 = standardize_code(code1)
                tokens_2 = standardize_code(code1)
                print("tokens_1:", tokens_1, "tokens_2:", tokens_2)
                context['search_res'] = [['Bilstm', 1]]
        elif load_template == 'deep_similarity.html':
            code1 = request.GET.get('code1', '')
            code2 = request.GET.get('code2', '')
            if code1 and code2:
                tokens_1 = ','.join(standardize_code(code1))
                tokens_2 = ','.join(standardize_code(code1))
                dis = edit_distance(tokens_1, tokens_2)
                print(tokens_1, tokens_2, dis)
                context['search_res'] = [['edit_distance', dis]]
        elif load_template == 'trans_similarity.html':
            code1 = request.GET.get('code1', '')
            code2 = request.GET.get('code2', '')
            print("code1:", code1, "code2:", code2)
            if code1 and code2:
                tokens_1 = standardize_code(code1)
                tokens_2 = standardize_code(code1)
                print("tokens_1:", tokens_1, "tokens_2:", tokens_2)
                dis = jaccard_similarity(set(tokens_1), set(tokens_2))
                context['search_res'] = [['jaccard', dis]]

        # context['search_res'] = [[]]

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:
        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except Exception as e:
        print(e)
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))


def refreshData(request):
    executor.submit(longtimeFun)
    return HttpResponse("后台更新中，预期3-10分钟")


def longtimeFun():

    print("全部完成")

# 1050537,1050869,1050933,1050907
# http://127.0.0.1:8000/form_elements.html
