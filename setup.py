#!/usr/bin/env python3
"""
强化学习量化交易系统安装脚本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# 读取requirements文件
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="rl-trading-system",
    version="0.1.0",
    author="RL Trading Team",
    author_email="team@rltrading.com",
    description="基于强化学习与Transformer的A股量化交易智能体系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rl-trading/rl-trading-system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-trading-train=scripts.train:main",
            "rl-trading-evaluate=scripts.evaluate:main",
            "rl-trading-deploy=scripts.deploy:main",
            "rl-trading-monitor=scripts.monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rl_trading_system": ["config/*.yaml"],
    },
    zip_safe=False,
)