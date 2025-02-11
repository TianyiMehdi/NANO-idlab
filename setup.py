from setuptools import setup, find_packages

# 读取 README 文件作为 long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NANO-filter",  # 包名
    version="1.0.0",  # 版本号
    license="MIT",
    author="Tianyi Zhang",
    author_email="zhangtia24@mails.tsinghua.edu.cn",
    description="Nonlinear Bayesian Filtering with Natural Gradient Gaussian Approximation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TianyiMehdi/NANO-filter",  # GitHub仓库地址
    packages=find_packages(),  # 自动查找包
    install_requires=[
        "autograd==1.6.2",
        "dataclasses_json==0.6.7",
        "einops==0.8.0",
        "filterpy==1.4.5",
        "matplotlib==3.8.2",
        "numpy==2.1.2",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "tqdm==4.66.1",
    ],  # 安装依赖
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 更改为你的许可证
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Python版本要求
)
